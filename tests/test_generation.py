import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import rag_generation as g

CHUNKS = [{"document_id": "DOC-002", "section": "Contamination", "text": "Contamination is foreign material."}]


def test_model_name_normalization():
    assert g.normalize_model_name("models/gemini-flash-latest") == "gemini-flash-latest"
    assert g.normalize_model_name(" gemini-x ") == "gemini-x"


def test_no_client_returns_labelled_fallback():
    r = g.generate_answer("q", CHUNKS, client=None)
    assert r["used_fallback"] and r["fallback_reason"] == "no_api_key" and r["provider"] is None
    assert "TEMPLATE FALLBACK" in r["answer"]


class _FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"models": [{"name": "llama3.2:3b"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send(200, {"message": {"content": "Foreign material. Source: DOC-002"}, "done_reason": "stop"})


@pytest.fixture
def fake_ollama(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{srv.server_port}")
    yield srv
    srv.shutdown()


def test_ollama_path_and_auto_selection(fake_ollama, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = g.get_llm_client()
    assert g.llm_provider_name(client) == "ollama"
    r = g.generate_answer("What is contamination?", CHUNKS, client=client)
    assert not r["used_fallback"] and r["provider"] == "ollama" and "DOC-002" in r["answer"]


def test_ollama_not_running_is_labelled(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")
    r = g.generate_answer("q", CHUNKS, client=g.OllamaClient())
    assert r["used_fallback"] and r["fallback_reason"] == "ollama_not_running"


def test_gemini_429_retries_then_succeeds(monkeypatch):
    from google.genai import errors

    class E(errors.ClientError):
        def __init__(self, code, msg):
            Exception.__init__(self, msg)
            self.code, self.message = code, msg

    calls = []

    class Models:
        def generate_content(self, **kw):
            calls.append(1)
            if len(calls) < 3:
                raise E(429, "rate")

            class R:
                text, candidates = "ok", []
            return R()

    class Client:
        models = Models()

    monkeypatch.setattr(time, "sleep", lambda s: None)
    r = g.generate_answer("q", CHUNKS, client=Client())
    assert r["answer"] == "ok" and len(calls) == 3 and r["provider"] == "gemini"


def test_gemini_daily_quota_does_not_retry(monkeypatch):
    from google.genai import errors

    class E(errors.ClientError):
        def __init__(self, code, msg):
            Exception.__init__(self, msg)
            self.code, self.message = code, msg

    calls = []

    class Models:
        def generate_content(self, **kw):
            calls.append(1)
            raise E(429, "Quota exceeded: GenerateRequestsPerDay")

    class Client:
        models = Models()

    r = g.generate_answer("q", CHUNKS, client=Client())
    assert r["fallback_reason"] == "rate_limit_or_quota" and len(calls) == 1


def test_gemini_404_is_model_not_found():
    from google.genai import errors

    class E(errors.ClientError):
        def __init__(self):
            Exception.__init__(self, "nf")
            self.code, self.message = 404, "nf"

    class Models:
        def generate_content(self, **kw):
            raise E()

    class Client:
        models = Models()

    assert g.generate_answer("q", CHUNKS, client=Client())["fallback_reason"] == "model_not_found"
