"""End-to-end API test. Needs the exported models (run scripts/export_inference_models.py first)."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("chromadb")
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "resnet_ocsvm_bundle.joblib").exists(),
                                reason="exported models not found")


def _png(brightness: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(brightness, 15, (96, 96, 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["MONITORING_LOG"] = str(tmp_path_factory.mktemp("logs") / "p.jsonl")
    os.environ["LLM_PROVIDER"] = "gemini"      # no key in tests -> deterministic labelled fallback
    os.environ.pop("GEMINI_API_KEY", None)
    import api
    import monitoring
    with TestClient(api.app) as c:
        # The synthetic export has a tiny validation set, so give the endpoint a realistic reference
        # (200 normal-looking images, brightness ~120) to test the drift logic against.
        rng = np.random.default_rng(0)
        api.state["monitor_reference"] = monitoring.build_reference(
            rng.normal(0, 0.1, 200), rng.normal(120, 15, 200), rng.normal(15, 2, 200),
            api.state["model_version"], "test reference")
        yield c


def test_health_reports_version_and_provider(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["anomaly_detector_loaded"] and h["model_version"].startswith("sha256:")
    assert h["llm_provider"] == "none"


def test_predict_logs_and_monitoring_endpoint(client):
    for i in range(3):
        r = client.post("/predict", files={"file": ("a.png", _png(120, i), "image/png")})
        assert r.status_code == 200 and r.json()["label"] in ("Normal", "Defective")
    mon = client.get("/monitoring").json()
    assert mon["summary"]["total_predictions"] == 3
    assert mon["drift"]["status"] == "insufficient_data"
    assert mon["model_version"].startswith("sha256:")
    assert len(mon["summary"]["model_versions"]) == 1


def test_drift_flagged_when_inputs_change(client):
    for i in range(40):  # very bright images vs. the reference images
        client.post("/predict", files={"file": ("b.png", _png(250, 100 + i), "image/png")})
    drift = client.get("/monitoring").json()["drift"]
    assert drift["features"]["brightness"]["status"] == "drift", drift
    assert drift["status"] == "drift"


def test_ask_returns_labelled_fallback_without_llm(client):
    r = client.post("/ask", json={"question": "What is contamination?"}).json()
    assert r["used_fallback"] and r["provider"] is None and r["sources"]


def test_bad_upload_is_rejected(client):
    assert client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
