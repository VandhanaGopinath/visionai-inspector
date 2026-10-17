"""
rag_generation.py
==================
Calls Google's Gemini API to answer a question using ONLY the retrieved
chunks as context — the generation half of retrieval-augmented generation.
Requires a GEMINI_API_KEY environment variable (create one at Google AI
Studio; this code only ever reads it from the environment — it is never
hardcoded, logged, or printed).

This replaced an earlier Anthropic Claude-based implementation specifically
to use Gemini's free tier instead of a paid API. The public interface
(`get_gemini_client()`, `generate_answer(question, retrieved_chunks,
client=...)`) mirrors the old `get_anthropic_client()` /
`generate_answer(..., client=anthropic_client)` shape on purpose, so the
notebook needed minimal changes beyond the client variable name.

Two LLM providers sit behind the same `generate_answer()` call:
  * Gemini (cloud, free tier, needs GEMINI_API_KEY)
  * Ollama (local, no key, no quota, no internet at question time)
`get_llm_client()` picks one (LLM_PROVIDER = auto | gemini | ollama) and
`generate_answer()` dispatches on the client type, so callers don't change.

The system prompt is the actual hallucination-resistance mechanism: it
explicitly instructs Gemini to decline rather than guess when the
retrieved context doesn't contain the answer. This is directly testable
against the project's evaluation set (the 9 questions specifically
designed to have no answer in the documents).
"""

import json
import os
import time
import urllib.error
import urllib.request
import warnings

# The SDK emits an informational warning recommending Chat.send_message over
# direct generate_content for function-calling use cases -- irrelevant here
# since this project doesn't use tool/function calling, and it would
# otherwise print once per question across the 30-question evaluation loop.
warnings.filterwarnings("ignore", message="Direct use of automatic function calling.*")


def _load_dotenv_if_available():
    """
    Load a local, gitignored `.env` file (project root) into the environment
    so GEMINI_API_KEY / GEMINI_MODEL work no matter which terminal launches
    the app. Existing environment variables win. Never prints values.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"), override=False)


_load_dotenv_if_available()


def normalize_model_name(name: str) -> str:
    """The google-genai SDK wants a bare name ('gemini-flash-latest'), not 'models/gemini-...'."""
    name = (name or "").strip()
    return name[len("models/"):] if name.startswith("models/") else name


# Gemini model names are retired regularly (gemini-2.0-flash was shut down), so this is
# configurable via the GEMINI_MODEL environment variable / .env instead of being hardcoded.
# `gemini-flash-latest` is Google's alias that always points at the current Flash model.
# Run `python scripts/list_gemini_models.py` to see exactly what YOUR key can use.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
GEMINI_MODEL = normalize_model_name(os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL)

SYSTEM_PROMPT = """You are a documentation assistant for an industrial bottle inspection system. \
Answer ONLY using the provided document excerpts below — never your own outside/general knowledge, \
and never information that isn't explicitly present in the excerpts. Do not invent information.

Audience: a factory floor operator or QA inspector, not an engineer. Write in plain, direct language \
they can act on immediately — short sentences, no jargon, no hedging filler.

Rules:
1. If the excerpts contain the answer, answer it directly and completely in 2-5 sentences (or a short \
numbered list for a procedure). Finish your thought before you cite anything — never end mid-sentence.
2. After your answer, on a new line, name which document(s) it came from (e.g. "Source: DOC-001, Inspection Procedure").
3. If the excerpts do NOT contain enough information to answer, say exactly: \
"The provided documentation does not contain enough information to answer this." \
Do not guess, estimate, or fill in a plausible-sounding number or procedure.
4. If multiple excerpts are relevant, synthesize across all of them into one coherent answer, not a list of quotes.
5. These documents are explicitly simulated/example documentation for a portfolio project, not a real \
industry standard — if asked whether this is official, say so plainly."""


def get_gemini_client():
    """
    Returns a configured google-genai Client if GEMINI_API_KEY is set, else
    None. Does not itself make a network call (an invalid/expired key
    isn't caught here — it surfaces on the first real generate_answer()
    call, where it's handled with a specific error message).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"WARNING: Gemini client could not be created ({type(e).__name__}: {e}).")
        return None


# ----------------------------------------------------------------- Ollama (local)
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"   # ~2 GB; any pulled chat model works via OLLAMA_MODEL
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaClient:
    """Minimal client for a local Ollama server's /api/chat (stdlib only -- no extra dependency)."""

    provider = "ollama"

    def __init__(self, host: str = None, model: str = None, timeout: float = 300.0):
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        # Generous: the first request has to load the model into memory (slow on CPU).
        self.timeout = timeout

    def _request(self, path: str, payload: dict = None, timeout: float = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def installed_models(self) -> list[str]:
        """Names of the models pulled into this Ollama server (raises URLError if it isn't running)."""
        tags = self._request("/api/tags", timeout=5)
        return [m.get("name", "") for m in tags.get("models", [])]

    def has_model(self) -> bool:
        wanted = self.model if ":" in self.model else self.model + ":latest"
        return wanted in self.installed_models()

    def chat(self, system: str, user: str) -> dict:
        return self._request("/api/chat", {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # temperature 0 = deterministic; num_ctx large enough for 4 retrieved chunks + prompt
            "options": {"temperature": 0.0, "num_predict": 1024, "num_ctx": 4096},
        })


def get_ollama_client():
    """Return an OllamaClient if the local server is running AND the model is pulled, else None."""
    client = OllamaClient()
    try:
        if client.has_model():
            return client
        print(f"WARNING: Ollama is running but model '{client.model}' is not pulled. "
              f"Run: ollama pull {client.model}")
    except (urllib.error.URLError, OSError, ValueError):
        pass  # server not running -- the normal case on a machine without Ollama
    return None


def get_llm_client():
    """
    Pick the answer-generation backend.
      LLM_PROVIDER=gemini  -> Gemini only        LLM_PROVIDER=ollama -> Ollama only
      LLM_PROVIDER=auto (default) -> Gemini if GEMINI_API_KEY is set, else local Ollama if available.
    Returns a client, or None (generate_answer then returns a clearly-labelled fallback).
    """
    provider = (os.environ.get("LLM_PROVIDER") or "auto").strip().lower()
    if provider == "gemini":
        return get_gemini_client()
    if provider == "ollama":
        return get_ollama_client()
    return get_gemini_client() or get_ollama_client()


def llm_provider_name(client) -> str:
    """'ollama', 'gemini' or 'none' -- for status displays and API responses."""
    if client is None:
        return "none"
    return getattr(client, "provider", "gemini")


def _is_daily_quota(e) -> bool:
    """True if a 429 is a per-DAY quota (waiting a minute won't help), judged from the error text."""
    text = str(e).lower()
    return "perday" in text or "per day" in text or "daily" in text


def _generate_with_retry(client, model, prompt, config, max_retries=3, base_wait=15):
    """
    Call Gemini, retrying only on per-minute rate limits (HTTP 429) and transient 503s,
    waiting 15s, 30s, 60s. Daily-quota 429s and all other errors are raised immediately.
    """
    from google.genai import errors
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except (errors.ClientError, errors.ServerError) as e:
            code = getattr(e, "code", None)
            retryable = code in (429, 503) and not _is_daily_quota(e)
            if not retryable or attempt == max_retries:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"Gemini HTTP {code} -- retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)


def format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunk records into a numbered, cited context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] ({chunk['document_id']} — {chunk['section']})\n{chunk['text']}")
    return "\n\n".join(parts)


def _fallback_result(reason: str, detail: str, question: str, retrieved_chunks: list[dict],
                     provider: str = "gemini") -> dict:
    """Build a clearly-labeled fallback response — never presented as if Gemini produced it."""
    doc_ids = ", ".join(sorted(set(c["document_id"] for c in retrieved_chunks))) or "none"
    answer = (
        f"[{provider.upper()} UNAVAILABLE ({reason}) — TEMPLATE FALLBACK, NOT A REAL ANSWER]\n"
        f"{detail}\n"
        f"Would have asked {provider.capitalize()} to answer '{question}' using {len(retrieved_chunks)} retrieved "
        f"chunk(s) from: {doc_ids}."
    )
    return {"answer": answer, "used_fallback": True, "fallback_reason": reason, "was_truncated": False,
            "provider": None}


def _generate_ollama(question: str, retrieved_chunks: list[dict], client: "OllamaClient") -> dict:
    """Answer with the local Ollama model. Same grounded prompt, same result shape as the Gemini path."""
    prompt = f"Document excerpts:\n\n{format_context(retrieved_chunks)}\n\nQuestion: {question}"
    try:
        data = client.chat(SYSTEM_PROMPT, prompt)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return _fallback_result(
                "ollama_model_missing",
                f"Ollama has no model '{client.model}' (HTTP 404). Run: ollama pull {client.model}",
                question, retrieved_chunks, "ollama")
        return _fallback_result("other_error", f"Ollama HTTP {e.code}", question, retrieved_chunks, "ollama")
    except (urllib.error.URLError, ConnectionError) as e:
        return _fallback_result(
            "ollama_not_running",
            f"Could not reach Ollama at {client.host} ({type(e).__name__}). Start it with `ollama serve` "
            "(the desktop app does this automatically).", question, retrieved_chunks, "ollama")
    except TimeoutError:
        return _fallback_result(
            "network_error", f"Ollama did not answer within {client.timeout:.0f}s (model too large for this machine?).",
            question, retrieved_chunks, "ollama")
    except Exception as e:
        return _fallback_result("other_error", f"{type(e).__name__}: {e}", question, retrieved_chunks, "ollama")

    answer_text = ((data.get("message") or {}).get("content") or "").strip()
    if not answer_text:
        return _fallback_result("empty_response", "Ollama returned an empty response.", question,
                                retrieved_chunks, "ollama")
    return {"answer": answer_text, "used_fallback": False, "fallback_reason": None,
            "was_truncated": data.get("done_reason") == "length", "provider": "ollama"}


def generate_answer(question: str, retrieved_chunks: list[dict], client=None, model: str = None) -> dict:
    """
    Generate a grounded answer to `question` using `retrieved_chunks` as the
    only source of information.

    Falls back to a clearly-labeled template response (no real LLM call,
    never presented as a genuine Gemini answer) if no client is available,
    or if the API call fails for any reason — with the specific reason
    distinguished so a real cause is visible rather than a silent failure:
    missing key, invalid/unauthorized key, rate limit / free-tier quota,
    network error, or any other API error.

    Returns
    -------
    dict with keys: answer, used_fallback (bool), fallback_reason (str or
    None), was_truncated (bool -- True if Gemini's response was cut off by
    the token limit before finishing, a real failure mode worth surfacing
    rather than silently showing a truncated answer as if it were complete)
    """
    if isinstance(client, OllamaClient):
        return _generate_ollama(question, retrieved_chunks, client)

    model = normalize_model_name(model or GEMINI_MODEL)
    if client is None:
        return _fallback_result(
            "no_api_key", "No LLM available: GEMINI_API_KEY is not set and no local Ollama model was found.", question, retrieved_chunks, "llm"
        )

    context = format_context(retrieved_chunks)
    prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

    try:
        from google.genai import types, errors

        response = _generate_with_retry(
            client, model, prompt,
            types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        answer_text = response.text
        if not answer_text:
            return _fallback_result(
                "empty_response", "Gemini returned an empty response (possibly blocked by safety filters).",
                question, retrieved_chunks
            )

        finish_reason = None
        if response.candidates:
            finish_reason = getattr(response.candidates[0], "finish_reason", None)
        was_truncated = finish_reason is not None and str(finish_reason).upper().endswith("MAX_TOKENS")

        return {"answer": answer_text, "used_fallback": False, "fallback_reason": None,
                "was_truncated": was_truncated, "provider": "gemini"}

    except errors.ClientError as e:
        code = getattr(e, "code", None)
        if code in (401, 403):
            return _fallback_result(
                "invalid_api_key", f"Gemini rejected the API key as invalid/unauthorized (HTTP {code}).",
                question, retrieved_chunks
            )
        if code == 404:
            return _fallback_result(
                "model_not_found",
                f"Model '{model}' is not available to this API key (HTTP 404) -- it may have been retired. "
                "Run `python scripts/list_gemini_models.py` and set GEMINI_MODEL to one of the listed names.",
                question, retrieved_chunks
            )
        if code == 429:
            return _fallback_result(
                "rate_limit_or_quota", "Gemini rate limit or free-tier quota exceeded (HTTP 429) even after retrying. If it is the daily "
                "quota, wait until it resets or switch GEMINI_MODEL to another model (quota is per model).",
                question, retrieved_chunks
            )
        return _fallback_result(
            "client_error", f"Gemini client-side error (HTTP {code}): {getattr(e, 'message', str(e))}",
            question, retrieved_chunks
        )

    except errors.ServerError as e:
        code = getattr(e, "code", None)
        return _fallback_result(
            "server_error", f"Gemini server-side error (HTTP {code}) — likely transient, try again.",
            question, retrieved_chunks
        )

    except Exception as e:
        # Covers network errors (e.g. httpx connection/timeout exceptions) and anything unforeseen --
        # broad on purpose so a bad connection or unexpected SDK exception never crashes the whole
        # evaluation loop over one question.
        error_name = type(e).__name__.lower()
        if "connect" in error_name or "timeout" in error_name or "network" in error_name:
            reason = "network_error"
        else:
            reason = "other_error"
        return _fallback_result(reason, f"{type(e).__name__}: {e}", question, retrieved_chunks)
