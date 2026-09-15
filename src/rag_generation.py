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

The system prompt is the actual hallucination-resistance mechanism: it
explicitly instructs Gemini to decline rather than guess when the
retrieved context doesn't contain the answer. This is directly testable
against the project's evaluation set (the 9 questions specifically
designed to have no answer in the documents).
"""

import os
import warnings

# The SDK emits an informational warning recommending Chat.send_message over
# direct generate_content for function-calling use cases -- irrelevant here
# since this project doesn't use tool/function calling, and it would
# otherwise print once per question across the 30-question evaluation loop.
warnings.filterwarnings("ignore", message="Direct use of automatic function calling.*")

GEMINI_MODEL = "gemini-2.0-flash"

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


def format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunk records into a numbered, cited context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] ({chunk['document_id']} — {chunk['section']})\n{chunk['text']}")
    return "\n\n".join(parts)


def _fallback_result(reason: str, detail: str, question: str, retrieved_chunks: list[dict]) -> dict:
    """Build a clearly-labeled fallback response — never presented as if Gemini produced it."""
    doc_ids = ", ".join(sorted(set(c["document_id"] for c in retrieved_chunks))) or "none"
    answer = (
        f"[GEMINI UNAVAILABLE ({reason}) — TEMPLATE FALLBACK, NOT A REAL ANSWER]\n"
        f"{detail}\n"
        f"Would have asked Gemini to answer '{question}' using {len(retrieved_chunks)} retrieved "
        f"chunk(s) from: {doc_ids}."
    )
    return {"answer": answer, "used_fallback": True, "fallback_reason": reason, "was_truncated": False}


def generate_answer(question: str, retrieved_chunks: list[dict], client=None, model: str = GEMINI_MODEL) -> dict:
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
    if client is None:
        return _fallback_result(
            "no_api_key", "No GEMINI_API_KEY found in the environment.", question, retrieved_chunks
        )

    context = format_context(retrieved_chunks)
    prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"

    try:
        from google.genai import types, errors

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
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

        return {"answer": answer_text, "used_fallback": False, "fallback_reason": None, "was_truncated": was_truncated}

    except errors.ClientError as e:
        code = getattr(e, "code", None)
        if code in (401, 403):
            return _fallback_result(
                "invalid_api_key", f"Gemini rejected the API key as invalid/unauthorized (HTTP {code}).",
                question, retrieved_chunks
            )
        if code == 429:
            return _fallback_result(
                "rate_limit_or_quota", "Gemini rate limit or free-tier quota exceeded (HTTP 429).",
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
