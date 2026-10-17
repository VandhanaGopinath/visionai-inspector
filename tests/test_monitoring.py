import numpy as np
import pandas as pd
from PIL import Image

import monitoring as m


def _ref(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return m.build_reference(rng.normal(0.5, 0.1, n), rng.normal(120, 10, n), rng.normal(40, 5, n), "v1", "test")


def _df(scores, bright, contrast, version="v1"):
    n = len(scores)
    return pd.DataFrame({"score": scores, "brightness": bright, "contrast": contrast, "model_version": version,
                         "label": ["Normal"] * n, "confidence": [0.9] * n, "latency_ms": [20.0] * n})


def test_psi_identical_is_near_zero_and_shift_is_large():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000)
    assert m.psi(a, rng.normal(0, 1, 2000)) < 0.1
    assert m.psi(a, rng.normal(2, 1, 2000)) > 0.25


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(2)
    rep = m.drift_report(_ref(), _df(rng.normal(0.5, 0.1, 100), rng.normal(120, 10, 100), rng.normal(40, 5, 100)), "v1")
    assert rep["status"] == "ok", rep


def test_drift_detected_when_images_get_brighter():
    rng = np.random.default_rng(3)
    rep = m.drift_report(_ref(), _df(rng.normal(0.5, 0.1, 100), rng.normal(170, 10, 100), rng.normal(40, 5, 100)), "v1")
    assert rep["status"] == "drift"
    assert rep["features"]["brightness"]["status"] == "drift"
    assert rep["features"]["score"]["status"] == "ok"


def test_insufficient_data_gives_no_verdict():
    rep = m.drift_report(_ref(), _df([0.5] * 5, [120] * 5, [40] * 5), "v1")
    assert rep["status"] == "insufficient_data"


def test_missing_reference_and_version_mismatch():
    df = _df(np.zeros(40), np.zeros(40), np.zeros(40))
    assert m.drift_report(None, df)["status"] == "no_reference"
    assert m.drift_report(_ref(), df, "other-version")["status"] == "reference_mismatch"


def test_logger_roundtrip_and_summary(tmp_path):
    log = m.PredictionLogger(tmp_path / "logs" / "p.jsonl", "v1")
    img = Image.new("RGB", (32, 32), (100, 100, 100))
    for i in range(5):
        log.log(source="test", label="Defective" if i < 2 else "Normal", score=0.1 * i, confidence=0.8,
                latency_ms=10 + i, image=img)
    (tmp_path / "logs" / "p.jsonl").open("a").write("not json\n")  # corrupt line must be skipped
    df = m.read_log(tmp_path / "logs" / "p.jsonl")
    assert len(df) == 5 and abs(df["brightness"].iloc[0] - 100) < 1
    s = m.summarize(df)
    assert s["total_predictions"] == 5 and s["defective_rate"] == 0.4 and s["model_versions"] == {"v1": 5}


def test_model_version_changes_with_file(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"a")
    v1 = m.compute_model_version(f)
    f.write_bytes(b"b")
    assert v1 != m.compute_model_version(f)
    assert m.compute_model_version(tmp_path / "missing") == "unknown"


def test_tiny_reference_gives_no_verdict():
    tiny = m.build_reference([0.1, 0.2], [100, 101], [30, 31], "v1", "test")
    rep = m.drift_report(tiny, _df(np.zeros(60), np.zeros(60), np.zeros(60)), "v1")
    assert rep["status"] == "reference_too_small"
