| variant | size_mb | latency_mean_ms | latency_p95_ms | peak_memory_mb | cosine_mean | max_abs_diff | roc_auc_test | decision_agreement_vs_pytorch |
|---|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 42.71 | 52.75 | 60.01 | 368.1 | 1.0 | 0.0 | 0.9698 | 1.0 |
| ONNX fp32 | 42.63 | 35.72 | 48.56 | 164.6 | 1.0 | 1e-05 | 0.9698 | 1.0 |
| ONNX int8 (static) | 10.77 | 15.01 | 17.74 | 71.1 | 0.999 | 0.5238 | 0.9913 | 0.9759 |
| ONNX int8 (dynamic) | 10.71 | 624.78 | 869.99 | 78.4 | 0.99741 | 0.68646 | 0.9079 | 0.9157 |

Measured on: Windows-10-10.0.22631-SP0, Intel64 Family 6 Model 140 Stepping 2, GenuineIntel, torch 2.14.0+cpu, onnxruntime 1.30.0. Latency = batch of 1, mean of 50 runs after warmup. Peak memory = fresh process incl. runtime import (compare ONNX variants with each other).
