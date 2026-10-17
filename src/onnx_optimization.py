"""
onnx_optimization.py
=====================
Stage 9 -- Edge AI optimization of the vision backbone (ResNet18 feature
extractor).

What problem this solves: a PyTorch model is great for research, but on an
edge device (a camera on a production line, a small CPU box) you want a
smaller, faster, framework-independent artifact. Two standard tools:

1. ONNX export   -- save the network as a framework-neutral graph that
                    ONNX Runtime executes with its own optimized CPU kernels.
2. Quantization  -- store weights (and compute activations) as 8-bit
                    integers instead of 32-bit floats: ~4x smaller weights
                    and often faster on CPUs with int8 support. The cost is
                    a small loss of numerical precision, so we MEASURE how
                    far the features drift, and how much the final anomaly
                    decision changes -- we never assume it is "fine".

Static quantization needs *calibration*: we run a few representative
images through the model so ONNX Runtime can learn the typical range of
every activation. Calibration images must come from the NORMAL TRAIN split
only, never the test set (that would leak test information into the model).

Nothing in this module fabricates numbers: every function either measures
something or transforms a file.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- export
def export_to_onnx(model, onnx_path: Path, input_size=(224, 224), opset: int = 17) -> Path:
    """
    Export a (frozen, eval-mode) torch model to ONNX with a dynamic batch axis.
    Input tensor is named "image" (N,3,H,W float32), output "features".
    """
    import torch

    model = model.cpu().eval()
    dummy = torch.randn(1, 3, *input_size)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["image"], output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        opset_version=opset,
    )
    with torch.no_grad():
        try:
            # The classic TorchScript exporter writes one self-contained file.
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kwargs)
        except TypeError:  # older torch without the `dynamo` argument
            torch.onnx.export(model, dummy, str(onnx_path), **kwargs)
    return onnx_path


# ------------------------------------------------------------------------ quantization
class _CalibrationReader:
    """Feeds calibration images to ONNX Runtime one at a time (implements its reader protocol)."""

    def __init__(self, images: np.ndarray, input_name: str = "image"):
        from onnxruntime.quantization import CalibrationDataReader

        # Subclass dynamically so importing this module doesn't require onnxruntime.quantization.
        class Reader(CalibrationDataReader):
            def __init__(inner):
                inner._it = iter([{input_name: images[i:i + 1].astype(np.float32)} for i in range(len(images))])

            def get_next(inner):
                return next(inner._it, None)

        self.reader = Reader()


def quantize_static_int8(fp32_path: Path, int8_path: Path, calibration_images: np.ndarray) -> Path:
    """
    Static post-training int8 quantization (QDQ format, per-channel int8
    weights, uint8 activations) calibrated on `calibration_images`
    (N,3,H,W float32, already preprocessed exactly like inference input).
    """
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod

    fp32_path, int8_path = Path(fp32_path), Path(int8_path)
    src = fp32_path
    # Shape inference / graph cleanup recommended before quantizing; skip quietly if unavailable.
    prepped = fp32_path.with_name(fp32_path.stem + "_prep.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(fp32_path), str(prepped), skip_symbolic_shape=True)
        src = prepped
    except Exception as e:  # noqa: BLE001 -- optional step
        print(f"  (quant_pre_process skipped: {type(e).__name__})")

    reader = _CalibrationReader(calibration_images).reader
    quantize_static(
        str(src), str(int8_path), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    if prepped.exists():
        prepped.unlink()
    return int8_path


def quantize_dynamic_int8(fp32_path: Path, out_path: Path) -> Path:
    """
    Dynamic int8 quantization: weights stored as int8, activation ranges
    computed on the fly (no calibration data). Simpler, but ONNX Runtime
    supports it best for Linear/MatMul layers (transformers), and it is
    often NOT a win for convolutional networks like ResNet -- included so
    the comparison shows that honestly rather than assuming it.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
    return Path(out_path)


# ------------------------------------------------------------------------ measurement
def file_size_mb(path: Path) -> float:
    """Size on disk, including an ONNX external-data sidecar file if one exists."""
    path = Path(path)
    total = path.stat().st_size
    for side in (path.with_suffix(path.suffix + ".data"), path.with_name(path.name + ".data")):
        if side.exists():
            total += side.stat().st_size
    return total / (1024 * 1024)


def _latency_stats(run_once, n_warmup: int, n_runs: int) -> dict:
    for _ in range(n_warmup):
        run_once()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_once()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95)), "n_runs": n_runs}


def make_session(onnx_path: Path, num_threads: int = 0):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if num_threads:
        opts.intra_op_num_threads = num_threads
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def benchmark_onnx_latency(onnx_path: Path, sample: np.ndarray, n_warmup: int = 10, n_runs: int = 50,
                           num_threads: int = 0) -> dict:
    """CPU latency of one batch-of-1 forward pass through an ONNX model."""
    sess = make_session(onnx_path, num_threads)
    name = sess.get_inputs()[0].name
    x = sample[:1].astype(np.float32)
    return _latency_stats(lambda: sess.run(None, {name: x}), n_warmup, n_runs)


def benchmark_torch_latency(model, sample: np.ndarray, n_warmup: int = 10, n_runs: int = 50) -> dict:
    """CPU latency of one batch-of-1 forward pass through the PyTorch model."""
    import torch

    model = model.cpu().eval()
    x = torch.from_numpy(sample[:1].astype(np.float32))

    def run():
        with torch.no_grad():
            model(x)

    return _latency_stats(run, n_warmup, n_runs)


_MEMORY_CHILD = r"""
import sys, json, numpy as np
kind, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
def peak_mb():
    # Linux: VmHWM = this process's own peak RSS (getrusage's ru_maxrss would inherit the
    # parent's peak across fork/exec and overstate it, so it is not used here).
    if sys.platform.startswith("linux"):
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    if sys.platform == "darwin":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    import psutil  # Windows
    return psutil.Process().memory_info().peak_wset / (1024 * 1024)
x = np.random.rand(1, 3, 224, 224).astype(np.float32)
if kind == "onnx":
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nm = s.get_inputs()[0].name
    for _ in range(n): s.run(None, {nm: x})
else:
    import torch
    from torchvision.models import resnet18
    m = resnet18(weights=None); m.fc = torch.nn.Identity(); m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
    t = torch.from_numpy(x)
    with torch.no_grad():
        for _ in range(n): m(t)
print(json.dumps({"peak_rss_mb": peak_mb()}))
"""


def measure_peak_memory_mb(kind: str, path: Path, n_inferences: int = 20) -> float | None:
    """
    Peak resident memory (MB) of a FRESH process that imports the runtime,
    loads the model and runs `n_inferences` forward passes. A fresh process
    is used so one variant's memory does not contaminate another's. The
    number includes the runtime's own import cost, so compare variants with
    the same runtime (the three ONNX ones) most directly.
    """
    try:
        out = subprocess.run([sys.executable, "-c", _MEMORY_CHILD, kind, str(path), str(n_inferences)],
                             capture_output=True, text=True, timeout=300, check=True)
        return float(json.loads(out.stdout.strip().splitlines()[-1])["peak_rss_mb"])
    except Exception as e:  # noqa: BLE001
        print(f"  (memory measurement failed for {path}: {type(e).__name__})")
        return None


def feature_fidelity(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """
    How closely do two feature matrices (N,D) agree? Cosine similarity per
    sample (1.0 = same direction) and the largest absolute element difference.
    """
    ref, cand = reference.astype(np.float64), candidate.astype(np.float64)
    cos = (ref * cand).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(cand, axis=1) + 1e-12)
    return {"cosine_mean": float(cos.mean()), "cosine_min": float(cos.min()),
            "max_abs_diff": float(np.abs(ref - cand).max())}


def onnx_features(onnx_path: Path, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Run an ONNX feature extractor over a (N,3,H,W) array, returning (N,512) features."""
    sess = make_session(onnx_path)
    name = sess.get_inputs()[0].name
    outs = [sess.run(None, {name: images[i:i + batch_size].astype(np.float32)})[0]
            for i in range(0, len(images), batch_size)]
    return np.concatenate(outs, axis=0)
