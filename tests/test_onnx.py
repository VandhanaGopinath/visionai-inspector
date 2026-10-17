import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import onnx_optimization as oo


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    # Big enough that int8 weight savings outweigh the fixed overhead of the quantization nodes.
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
    ).eval()


def test_export_matches_pytorch(tiny_model, tmp_path):
    path = oo.export_to_onnx(tiny_model, tmp_path / "m.onnx", input_size=(32, 32))
    x = np.random.rand(4, 3, 32, 32).astype(np.float32)
    with torch.no_grad():
        ref = tiny_model(torch.from_numpy(x)).numpy()
    out = oo.onnx_features(path, x, batch_size=3)  # different batch size also exercises the dynamic axis
    assert out.shape == ref.shape
    assert oo.feature_fidelity(ref, out)["cosine_min"] > 0.9999


def test_static_quantization_shrinks_and_stays_close(tiny_model, tmp_path):
    fp32 = oo.export_to_onnx(tiny_model, tmp_path / "m.onnx", input_size=(32, 32))
    calib = np.random.rand(10, 3, 32, 32).astype(np.float32)
    int8 = oo.quantize_static_int8(fp32, tmp_path / "m_int8.onnx", calib)
    assert oo.file_size_mb(int8) <= oo.file_size_mb(fp32)
    x = np.random.rand(6, 3, 32, 32).astype(np.float32)
    fid = oo.feature_fidelity(oo.onnx_features(fp32, x), oo.onnx_features(int8, x))
    assert fid["cosine_mean"] > 0.95


def test_latency_helpers_return_positive_numbers(tiny_model, tmp_path):
    path = oo.export_to_onnx(tiny_model, tmp_path / "m.onnx", input_size=(32, 32))
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    lat = oo.benchmark_onnx_latency(path, x, n_warmup=1, n_runs=3)
    assert lat["mean_ms"] > 0 and lat["p95_ms"] >= lat["p50_ms"]
