"""
benchmark_onnx.py  --  Stage 9: Edge AI optimization benchmark
================================================================
Exports the ResNet18 feature extractor to ONNX, quantizes it to int8, and
MEASURES (never assumes) how the variants compare on THIS machine:

    PyTorch fp32  |  ONNX fp32  |  ONNX int8 (static, calibrated)  |  ONNX int8 (dynamic)

For each variant: file size, CPU latency (mean / p50 / p95), peak memory of a
fresh process, feature fidelity vs. PyTorch fp32, and -- if the exported
anomaly detector exists -- the end-to-end effect on the real task
(ROC-AUC and Normal/Defective agreement on the untouched MVTec test set).

Calibration for static quantization uses NORMAL TRAIN images only.

Usage:
    python scripts/export_inference_models.py     # once, creates models/resnet_ocsvm_bundle.joblib
    python scripts/benchmark_onnx.py

Outputs: models/onnx/*.onnx and reports/stage9_onnx_benchmark.{json,md}
If the real MVTec data is missing the script falls back to synthetic noise
images and labels the whole report "SYNTHETIC -- smoke test only".
"""

import json
import platform
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_utils import build_protocol_splits, make_synthetic_placeholder_dataset
from pretrained_features import build_resnet_feature_extractor
from benchmarking import count_parameters, measure_model_size_mb
import onnx_optimization as oo

CATEGORY = "bottle"
RANDOM_SEED = 42
N_CALIBRATION = 50
N_LATENCY_RUNS = 50


def load_images(df: pd.DataFrame, preprocess) -> np.ndarray:
    from PIL import Image
    return np.stack([preprocess(Image.open(p).convert("RGB")).numpy() for p in df["filepath"]]).astype(np.float32)


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")  # edge-style CPU measurement, regardless of local GPU

    raw_root = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
    processed_root = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    onnx_dir = models_dir / "onnx"
    reports_dir = PROJECT_ROOT / "reports"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    category_path = raw_root / CATEGORY
    synthetic = not category_path.exists()
    if synthetic:
        print("Real MVTec data not found -> synthetic noise images. Report will be labelled SMOKE TEST ONLY.")
        make_synthetic_placeholder_dataset(raw_root, category=CATEGORY, seed=RANDOM_SEED)
    train_df, val_df, test_df = build_protocol_splits(category_path, val_frac=0.15, seed=RANDOM_SEED)
    print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)} (test used for evaluation only)")

    resnet, pretrained, preprocess = build_resnet_feature_extractor(device)
    print(f"ResNet18 pretrained weights: {pretrained}")

    calib = load_images(train_df.sample(min(N_CALIBRATION, len(train_df)), random_state=RANDOM_SEED), preprocess)
    test_images = load_images(test_df, preprocess)
    test_labels = (test_df["label"] == "defective").astype(int).to_numpy()

    # ------------------------------------------------------------------ build variants
    fp32_path = onnx_dir / "resnet18_fp32.onnx"
    int8_static_path = onnx_dir / "resnet18_int8_static.onnx"
    int8_dynamic_path = onnx_dir / "resnet18_int8_dynamic.onnx"
    torch_weights = onnx_dir / "resnet18_fp32_torch.pt"
    torch.save(resnet.state_dict(), torch_weights)

    print("\nExporting to ONNX ...")
    oo.export_to_onnx(resnet, fp32_path)
    print("Static int8 quantization (calibrating on normal train images) ...")
    oo.quantize_static_int8(fp32_path, int8_static_path, calib)
    variants = {"PyTorch fp32": None, "ONNX fp32": fp32_path, "ONNX int8 (static)": int8_static_path}
    try:
        print("Dynamic int8 quantization ...")
        oo.quantize_dynamic_int8(fp32_path, int8_dynamic_path)
        variants["ONNX int8 (dynamic)"] = int8_dynamic_path
    except Exception as e:  # noqa: BLE001
        print(f"  dynamic quantization failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ reference features / head
    with torch.no_grad():
        ref_feats = np.concatenate([resnet(torch.from_numpy(test_images[i:i + 16])).numpy()
                                    for i in range(0, len(test_images), 16)])
    bundle_path = models_dir / "resnet_ocsvm_bundle.joblib"
    bundle = joblib.load(bundle_path) if bundle_path.exists() else None
    if bundle is None:
        print("(no resnet_ocsvm_bundle.joblib -> skipping end-to-end ROC-AUC; run export_inference_models.py first)")

    def head_scores(feats):
        return bundle["ocsvm"].decision_function(bundle["scaler"].transform(feats))

    from sklearn.metrics import roc_auc_score
    ref_pred = None
    if bundle is not None:
        ref_scores = head_scores(ref_feats)
        ref_pred = (ref_scores < bundle["threshold"]).astype(int)  # 1 = Defective

    # ------------------------------------------------------------------ measure
    rows = []
    for name, path in variants.items():
        print(f"\nMeasuring: {name}")
        if path is None:
            lat = oo.benchmark_torch_latency(resnet, test_images, n_runs=N_LATENCY_RUNS)
            size = measure_model_size_mb(resnet)
            mem = oo.measure_peak_memory_mb("torch", torch_weights)
            feats = ref_feats
        else:
            lat = oo.benchmark_onnx_latency(path, test_images, n_runs=N_LATENCY_RUNS)
            size = oo.file_size_mb(path)
            mem = oo.measure_peak_memory_mb("onnx", path)
            feats = oo.onnx_features(path, test_images)

        row = {"variant": name, "size_mb": round(size, 2), "latency_mean_ms": round(lat["mean_ms"], 2),
               "latency_p50_ms": round(lat["p50_ms"], 2), "latency_p95_ms": round(lat["p95_ms"], 2),
               "peak_memory_mb": None if mem is None else round(mem, 1)}
        row.update({k: round(v, 5) for k, v in oo.feature_fidelity(ref_feats, feats).items()})
        if bundle is not None:
            scores = head_scores(feats)
            row["roc_auc_test"] = round(float(roc_auc_score(test_labels, -scores)), 4)
            pred = (scores < bundle["threshold"]).astype(int)
            row["decision_agreement_vs_pytorch"] = round(float((pred == ref_pred).mean()), 4)
            row["accuracy_at_threshold"] = round(float((pred == test_labels).mean()), 4)
        rows.append(row)
        print("  ", row)

    # ------------------------------------------------------------------ report
    base = rows[0]
    for r in rows:
        r["size_vs_pytorch"] = round(r["size_mb"] / base["size_mb"], 3)
        r["latency_vs_pytorch"] = round(r["latency_mean_ms"] / base["latency_mean_ms"], 3)

    meta = {
        "synthetic_data_smoke_test_only": synthetic,
        "resnet_pretrained_weights": pretrained,
        "parameters": count_parameters(resnet),
        "device": "CPU",
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": __import__("onnxruntime").__version__,
        "test_images": int(len(test_images)),
        "calibration_images": int(len(calib)),
        "latency_runs": N_LATENCY_RUNS,
    }
    (reports_dir / "stage9_onnx_benchmark.json").write_text(json.dumps({"meta": meta, "results": rows}, indent=2), encoding="utf-8")

    cols = ["variant", "size_mb", "latency_mean_ms", "latency_p95_ms", "peak_memory_mb", "cosine_mean", "max_abs_diff"]
    if bundle is not None:
        cols += ["roc_auc_test", "decision_agreement_vs_pytorch"]
    lines = []
    if synthetic or not pretrained:
        lines.append("> WARNING: " + ("synthetic noise images; " if synthetic else "") +
                     ("random-init ResNet18 (no pretrained weights); " if not pretrained else "") +
                     "these numbers are a pipeline SMOKE TEST, not project results.\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append(f"\nMeasured on: {meta['os']}, {meta['cpu']}, torch {meta['torch']}, onnxruntime {meta['onnxruntime']}. "
                 f"Latency = batch of 1, mean of {N_LATENCY_RUNS} runs after warmup. "
                 "Peak memory = fresh process incl. runtime import (compare ONNX variants with each other).")
    md = "\n".join(lines)
    (reports_dir / "stage9_onnx_benchmark.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    print(f"\nSaved -> {reports_dir / 'stage9_onnx_benchmark.md'} (+ .json)")


if __name__ == "__main__":
    main()
