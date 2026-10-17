"""
monitoring.py
==============
Stage 10 -- what happens AFTER a model is deployed.

A model that scored well on the test set can still get worse in production:
the camera is moved, lighting changes, a new bottle batch looks different.
Nobody tells you -- there are no labels in production. So we watch proxies:

1. Operational health   latency, request counts, errors
2. Prediction health    share of Defective predictions, confidence levels
3. Input drift          has the *input data* changed compared to what the
                        model was calibrated on? (image brightness/contrast)
4. Score drift          has the distribution of anomaly scores shifted
                        compared to normal validation images?
5. Model version        every logged prediction records which model made it

Drift is measured by comparing production values to a REFERENCE saved when
the model was exported (normal validation images only):

* PSI (Population Stability Index): bins the reference into quantiles and
  measures how much production mass moved between bins.
  Common rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 major.
* KS test (Kolmogorov-Smirnov): is the difference between the two
  distributions larger than chance would explain? (small p-value = shifted)

Honest limits: with no ground-truth labels in production this can only say
"the data / scores look different from before", not "accuracy dropped". A
real deployment would also sample some bottles for human review to measure
true accuracy. Small samples make both tests noisy, so no verdict is given
below MIN_SAMPLES records.

Pure numpy / scipy / pandas -- no torch needed, so it is cheap to import.
"""

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MIN_SAMPLES = 30
MIN_REFERENCE = 20
PSI_WARNING = 0.10
PSI_DRIFT = 0.25
KS_P_WARNING = 0.05
KS_P_DRIFT = 0.01
DRIFT_FEATURES = ("score", "brightness", "contrast")


# ------------------------------------------------------------------ helpers
def compute_model_version(path: Path) -> str:
    """Short content hash of the model file: changes whenever the model is retrained/replaced."""
    path = Path(path)
    if not path.exists():
        return "unknown"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def image_stats(image) -> dict:
    """Brightness (mean grey level, 0-255), contrast (std of grey level) and size of a PIL image."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    return {"brightness": float(grey.mean()), "contrast": float(grey.std()),
            "width": int(image.width), "height": int(image.height)}


def build_reference(scores, brightness, contrast, model_version: str, source: str) -> dict:
    """Package the reference distributions (from normal validation images) for drift comparison."""
    return {
        "model_version": model_version,
        "source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score": [float(v) for v in scores],
        "brightness": [float(v) for v in brightness],
        "contrast": [float(v) for v in contrast],
    }


def save_reference(reference: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference), encoding="utf-8")


def load_reference(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ logging
class PredictionLogger:
    """Thread-safe append-only JSON-lines log of every prediction (one line per request)."""

    def __init__(self, path: Path, model_version: str = "unknown"):
        self.path = Path(path)
        self.model_version = model_version
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, source: str, label: str, score: float, confidence: float, latency_ms: float,
            image=None, extra: dict | None = None) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,                      # "api" / "streamlit"
            "model_version": self.model_version,
            "label": label,
            "score": float(score),
            "confidence": float(confidence),
            "latency_ms": float(latency_ms),
        }
        if image is not None:
            record.update(image_stats(image))
        if extra:
            record.update(extra)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record


def read_log(path: Path, last_n: int | None = None) -> pd.DataFrame:
    """Load the prediction log (empty DataFrame if there is none yet). Skips corrupt lines."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    if last_n and len(df) > last_n:
        df = df.tail(last_n).reset_index(drop=True)
    return df


# ------------------------------------------------------------------ drift statistics
def psi(reference, current, bins: int | None = None, eps: float = 1e-4) -> float:
    """
    Population Stability Index of `current` relative to `reference`.
    Bin edges are reference quantiles, widened to +/- infinity so production
    values outside the reference range still land in an end bin.

    PSI is biased upward on small samples (sparse bins look "shifted" by
    chance), so by default the number of bins adapts: about 20 current
    samples per bin, between 3 and 10 bins.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if bins is None:
        bins = int(np.clip(len(current) // 20, 3, 10))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # (almost) constant reference: fall back to equal-width
        lo, hi = reference.min(), reference.max()
        edges = np.linspace(lo - 1e-6, hi + 1e-6, 3)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_frac = np.histogram(current, bins=edges)[0] / len(current)
    ref_frac, cur_frac = np.clip(ref_frac, eps, None), np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def drift_status(psi_value: float, ks_p: float, n_tests: int = len(DRIFT_FEATURES)) -> str:
    """Both measures must agree, which keeps false alarms down: PSI says how big the shift is,
    the KS test says whether it is bigger than chance. Because several features are tested at once,
    the KS p-value limits are divided by the number of tests (Bonferroni correction)."""
    if psi_value >= PSI_DRIFT and ks_p < KS_P_DRIFT / n_tests:
        return "drift"
    if psi_value >= PSI_WARNING and ks_p < KS_P_WARNING / n_tests:
        return "warning"
    return "ok"


def drift_report(reference: dict | None, df: pd.DataFrame, model_version: str | None = None) -> dict:
    """
    Compare logged production values to the reference distributions.

    Only rows from `model_version` are compared (a retrained model has a new
    reference, so mixing versions would be meaningless). Returns a dict:
    {status, n_samples, model_version, features: {name: {psi, ks_p, status, ...}}, note}.
    """
    if reference is None:
        return {"status": "no_reference", "n_samples": int(len(df)), "features": {},
                "note": "No monitoring_reference.json found. Run scripts/export_inference_models.py."}
    if model_version and "model_version" in df.columns:
        df = df[df["model_version"] == model_version]
    if reference.get("model_version") and model_version and reference["model_version"] != model_version:
        return {"status": "reference_mismatch", "n_samples": int(len(df)), "features": {},
                "note": "The reference was built for a different model version. Re-run the export script."}
    if len(reference.get("score", [])) < MIN_REFERENCE:
        return {"status": "reference_too_small", "n_samples": int(len(df)), "features": {},
                "note": f"The drift reference has only {len(reference.get('score', []))} images "
                        f"(need {MIN_REFERENCE}+). Re-run the export script on the real dataset."}
    n = int(len(df))
    if n < MIN_SAMPLES:
        return {"status": "insufficient_data", "n_samples": n, "features": {},
                "note": f"Need at least {MIN_SAMPLES} logged predictions for a drift verdict (have {n})."}

    features = {}
    for name in DRIFT_FEATURES:
        if name not in df.columns or name not in reference:
            continue
        cur = df[name].dropna().to_numpy(dtype=float)
        ref = np.asarray(reference[name], dtype=float)
        if len(cur) < MIN_SAMPLES or len(ref) < 2:
            continue
        p = psi(ref, cur)
        ks = stats.ks_2samp(ref, cur)
        features[name] = {"psi": round(p, 4), "ks_statistic": round(float(ks.statistic), 4),
                          "ks_p_value": round(float(ks.pvalue), 6), "status": drift_status(p, float(ks.pvalue)),
                          "reference_mean": round(float(ref.mean()), 4), "current_mean": round(float(cur.mean()), 4)}
    order = {"ok": 0, "warning": 1, "drift": 2}
    overall = max((f["status"] for f in features.values()), key=order.get, default="ok")
    return {"status": overall, "n_samples": n, "model_version": model_version, "features": features,
            "note": "Compared with NORMAL validation images. Without production labels this shows data/score "
                    "change, not accuracy loss."}


# ------------------------------------------------------------------ summary
def summarize(df: pd.DataFrame) -> dict:
    """Operational + prediction-health summary of the logged predictions."""
    if df.empty:
        return {"total_predictions": 0}
    out = {
        "total_predictions": int(len(df)),
        "first_prediction": str(df["ts"].iloc[0]) if "ts" in df else None,
        "last_prediction": str(df["ts"].iloc[-1]) if "ts" in df else None,
        "prediction_distribution": {k: int(v) for k, v in df["label"].value_counts().items()},
        "defective_rate": round(float((df["label"] == "Defective").mean()), 4),
        "confidence_mean": round(float(df["confidence"].mean()), 4),
        "confidence_p10": round(float(df["confidence"].quantile(0.10)), 4),
        "latency_ms": {"mean": round(float(df["latency_ms"].mean()), 2),
                       "p50": round(float(df["latency_ms"].quantile(0.5)), 2),
                       "p95": round(float(df["latency_ms"].quantile(0.95)), 2)},
    }
    if "model_version" in df:
        out["model_versions"] = {k: int(v) for k, v in df["model_version"].value_counts().items()}
    return out
