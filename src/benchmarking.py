"""
benchmarking.py
================
Measure the concrete numbers a "CNN vs ViT" comparison actually needs:
how many parameters a model has, how much disk space its weights take, and
how long a forward pass takes. Accuracy alone doesn't tell you whether a
model is practical to deploy — this is exactly the kind of measurement
Stage 9 (Edge AI optimization) will care about too, just introduced here
where it's first relevant.
"""

import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Total number of learnable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def measure_model_size_mb(model: nn.Module) -> float:
    """
    Serialize a model's weights to a temporary file and measure the size on
    disk in megabytes — a direct, honest measurement rather than an
    estimate from parameter count and dtype (which would miss real-world
    overhead from how PyTorch serializes state dicts).
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    torch.save(model.state_dict(), tmp_path)
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    tmp_path.unlink()
    return size_mb


@torch.no_grad()
def measure_inference_latency(model: nn.Module, input_tensor: torch.Tensor, device: torch.device,
                               n_warmup: int = 5, n_runs: int = 20) -> float:
    """
    Average wall-clock time (milliseconds) for one forward pass of
    `input_tensor` through `model`, run n_runs times after n_warmup
    untimed warmup passes (the first few runs are typically slower due to
    lazy initialization / caching effects, so we exclude them from the
    average to get a representative steady-state number).

    Measured on whatever `device` actually is (CPU here, unless a GPU is
    available) — always report which device a latency number came from,
    since the same model can be 10-50x faster on a GPU.
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    for _ in range(n_warmup):
        model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / n_runs * 1000  # ms per forward pass
