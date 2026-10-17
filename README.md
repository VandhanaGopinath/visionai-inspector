# VisionAI Inspector

An end-to-end AI/ML portfolio project: **industrial bottle inspection** with computer vision, a RAG-based
assistant that answers from inspection documentation, a REST API, a web app, edge optimization (ONNX +
int8), monitoring with drift detection, and Docker packaging. Built stage by stage on the public
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) `bottle` category.

```
Image -> anomaly detector (ResNet18 + One-Class SVM) -> Normal / Defective + score
      -> localization (autoencoder error map) -> bounding box + heatmap
      -> RAG (Sentence Transformers + ChromaDB + Gemini or local Ollama) -> grounded operator guidance
      -> every prediction logged -> drift / health monitoring
```

## Status

| Stage | What | State |
|---|---|---|
| 1 | Data analysis and protocol-faithful split | done |
| 2 | Classical ML anomaly detection (OCSVM / Isolation Forest) | done |
| 3 | Deep learning (autoencoder, pretrained ResNet18) | done |
| 4 | Localization / segmentation (IoU, Dice) | done |
| 5 | Vision Transformer vs CNN comparison | done |
| 6 | RAG + LLM (Gemini free tier or local Ollama) | done |
| 7 | Vision-language model | **not built** (intentionally out of scope) |
| 8 | FastAPI backend (`/predict /detect /ask /health /metrics /monitoring`) | done |
| 9 | Edge optimization: ONNX + int8 quantization | done, benchmark script included |
| 10 | Monitoring: logging, health summary, PSI/KS drift | done |
| 11 | Docker + deployment | files written and validated; image build not run by me (see below) |

Known limits, stated plainly:
- The system answers **Normal vs. Defective** only. It does not classify the defect type.
- The knowledge-base documents are simulated, written for this project. They are not a real standard.
- Numbers in the stage sections come from notebook runs. Anything marked *smoke test* was produced on
  synthetic data or random weights to check the code, and is not a result.
- Monitoring without production labels can show that inputs or scores **changed**, not that accuracy **dropped**.
- Small local LLMs (Ollama) follow the "refuse if not in the documents" rule less reliably than Gemini.
  Re-run the notebook 06 evaluation with your provider and read the refusal rate before relying on it.

## Quick start

```bash
pip install -r requirements.txt            # same on Windows PowerShell
python scripts/export_inference_models.py  # fits + saves models/ (needs the dataset in data/raw/)
streamlit run app.py                       # web app on http://localhost:8501
uvicorn api:app --reload                   # API on http://localhost:8000/docs
python -m pytest tests -q                  # tests (pip install -r requirements-dev.txt first)
```
Answers need either `GEMINI_API_KEY` in `.env` or a local Ollama (see the Gemini and Ollama sections below).

---

# Stage 1: Data Analysis & Preprocessing

## Dataset choice: MVTec AD (category: `bottle`)

**What it is:** MVTec AD is a real-world industrial anomaly-detection benchmark
released by MVTec Software GmbH. It contains high-resolution photographs of 15
object/texture categories (bottle, cable, capsule, hazelnut, screw, metal_nut,
etc.), each with:

- A `train/good/` split — defect-free images only.
- A `test/` split with a `good/` subfolder plus one subfolder per defect type
  (e.g. `broken_large/`, `broken_small/`, `contamination/`).
- A `ground_truth/` folder with **pixel-level binary masks** for every
  defective test image.

**Confirmed official counts for `bottle`** (from the original Bergmann et al.
2019 paper and independent secondary sources — the notebook checks our own
downloaded copy against these in section 3b):

| Split | Folder | Count |
|---|---|---|
| `train` | `good` | 209 |
| `test` | `good` | 20 |
| `test` | defective (broken_large + broken_small + contamination, combined) | 63 |
| `ground_truth` | masks | 63 |
| **Total** | | **292** |

**Why `train` is almost entirely defect-free — and why this matters for how
we split the data:** MVTec AD is built for *unsupervised / semi-supervised
anomaly detection*, not standard supervised classification. This mirrors a
real constraint in industrial QC: normal units are abundant, but defects are
rare and highly varied, so exhaustively labeled defect data for training is
often impractical. The standard approach instead models what "normal" looks
like and flags deviations at test time — which is exactly why `train`
contains only `good` images and `test` mixes in every defect type. Any split
of this data that mixes `test` images into training breaks that protocol and
makes results incomparable to published benchmarks (see section 11 of the
notebook for the full explanation and the corrected split logic).

**Why it fits this project specifically (not just "a defect dataset"):**

1. It's real industrial photography, not a toy/synthetic set — matches the
   problem statement (industrial component inspection).
2. It already carries every label the final architecture needs at once:
   binary normal/defective (Stage 2–3 classification), defect *type*
   (multi-class classification), and pixel-level masks (Stage 4 segmentation,
   and later "where is the defect located" for the VLM/RAG stage). We won't
   need to switch datasets as the project grows.
3. Its train/test split is intentionally *not* a random split — `train/`
   contains only normal parts. That's a deliberate anomaly-detection design
   choice, and reasoning about how to turn that into a proper
   train/val/test split for supervised classification (Stage 1's job) is
   itself a useful, realistic data-engineering exercise.
4. Any single category is small (bottle: ~290 images, ~110 MB), so Stage 1
   stays lightweight, per the "don't over-engineer the first stage" rule.
   Later stages can add more categories using the exact same code, since the
   folder layout is identical across all 15.
5. It's the standard benchmark used in industrial-inspection papers (PaDiM,
   PatchCore, anomalib, etc.) — being able to say "I used MVTec AD and
   understand why" carries real weight in an interview.

We start with **one category, `bottle`**, to keep Stage 1 focused. The
notebook is written so that switching `CATEGORY = "bottle"` to another
category, or looping over several, is a one-line change later.

## Getting the data (manual step — required)

This sandbox cannot reach `mvtec.com` or `kaggle.com` directly (network is
restricted to package registries), so **you need to download the data
yourself, once, on your own machine**:

1. Official source (free, requires filling a short access form):
   https://www.mvtec.com/company/research/datasets/mvtec-ad
   Download the full archive or, if offered, just the `bottle.tar.xz`
   category file.
2. Alternative: search "mvtec anomaly detection" or "mvtec-ad" on Kaggle —
   several mirrors exist that let you download without the form, e.g. via
   `kaggle datasets download <mirror-name>`.
3. Extract it so you end up with this structure:

```
visionai-inspector/
└── data/
    └── raw/
        └── mvtec_ad/
            └── bottle/
                ├── train/
                │   └── good/                *.png
                ├── test/
                │   ├── good/                 *.png
                │   ├── broken_large/         *.png
                │   ├── broken_small/         *.png
                │   └── contamination/        *.png
                └── ground_truth/
                    ├── broken_large/         *_mask.png
                    ├── broken_small/         *_mask.png
                    └── contamination/        *_mask.png
```

If you download more categories later, just drop each one alongside
`bottle/` under `data/raw/mvtec_ad/` — the code discovers categories
automatically.

## Running Stage 1

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_stage1.ipynb
```

The notebook (`notebooks/01_eda_stage1.ipynb`) will:

- Auto-detect whether real MVTec data is present in `data/raw/mvtec_ad/`.
- If it's **not** present yet, generate a small synthetic placeholder
  dataset with the same folder layout, purely so every cell in the
  notebook is verified to run end-to-end. Every plot/statistic produced
  from placeholder data is clearly labeled `[SYNTHETIC PLACEHOLDER DATA]`
  in its title — these numbers are not real results and must not be
  quoted anywhere (portfolio, resume, interview).
- If real data **is** present, check the on-disk counts against the
  official reference counts above (section 3b) before trusting any
  downstream statistic, and print `OK`/`MISMATCH` per folder.
- Once you drop the real `bottle/` folder into place and re-run, all the
  same cells produce genuine statistics from the real dataset.
- Build a **protocol-preserving** split (section 11): `train`/`val` are
  carved only from MVTec's original normal-only `train/good` pool (roughly
  85/15), and the original `test` set — every `good` and every defective
  image — is left completely untouched, used only for final evaluation.
  This deliberately does **not** mix defective images into training, and
  does **not** touch `test` at all, unlike a naive random/stratified split
  across the whole dataset, which would break MVTec's anomaly-detection
  protocol and leak official test images into training.

## Project layout

```
visionai-inspector/
├── data/
│   ├── raw/             # original downloaded images (not committed to git)
│   ├── processed/        # resized/normalized arrays + split CSVs produced by the notebooks
│   └── knowledge_base/   # RAG source documents (Stage 6) — committed to git, these are yours
├── notebooks/
│   ├── 01_eda_stage1.ipynb
│   ├── 02_baseline_anomaly_detection.ipynb
│   ├── 03_deep_learning_anomaly_detection.ipynb
│   ├── 04_defect_localization.ipynb
│   ├── 05_vision_transformers.ipynb
│   ├── 06_rag_llm.ipynb
│   ├── 09_edge_optimization.ipynb   # reads the Stage 9 benchmark report
│   └── 10_monitoring.ipynb          # PSI / KS drift explained with a simulation
├── src/
│   ├── data_utils.py          # reusable, documented loading/preprocessing functions
│   ├── features.py             # hand-crafted feature extraction (Stage 2)
│   ├── evaluation.py           # shared anomaly-detection evaluation logic (Stage 2+3+4+5)
│   ├── torch_dataset.py        # PyTorch Dataset wrapper (Stage 3)
│   ├── autoencoder.py          # custom convolutional autoencoder (Stage 3, Model A)
│   ├── pretrained_features.py  # frozen pretrained ResNet18 + ViT-B/16 feature extractors (Stage 3+4+5)
│   ├── masks.py                 # ground-truth mask loading + IoU/Dice/bbox utilities (Stage 4)
│   ├── benchmarking.py          # parameter count / model size / inference latency (Stage 5)
│   ├── rag_documents.py         # knowledge base parsing + evaluation question parsing (Stage 6)
│   ├── rag_embeddings.py        # local Sentence Transformer embeddings, with offline fallback (Stage 6)
│   ├── rag_generation.py        # grounded generation: Gemini or local Ollama (Stage 6)
│   ├── onnx_optimization.py     # ONNX export, int8 quantization, size/latency/memory measurement (Stage 9)
│   ├── monitoring.py            # prediction log, PSI/KS drift, health summary (Stage 10)
│   └── inference.py             # single-image inference wrappers for the app (reuses Stage 3+4 logic)
├── scripts/
│   ├── export_inference_models.py  # one-time: fits + saves models/ artifacts + drift reference
│   ├── benchmark_onnx.py           # Stage 9: ONNX + int8 benchmark -> reports/
│   └── list_gemini_models.py       # which Gemini models your key can use
├── tests/                       # pytest: monitoring, LLM providers, ONNX, API
├── app.py                       # Streamlit web interface (inspect, ask, monitoring)
├── api.py                       # FastAPI backend (same models, callable by anything)
├── Dockerfile, docker-compose.yml, .dockerignore   # Stage 11
├── .github/workflows/ci.yml     # runs the tests on every push
├── DEPLOYMENT.md
├── requirements.txt, requirements-dev.txt
└── README.md
```

Add a `.gitignore` entry for `data/raw/` and `data/processed/` before
committing to a repo — the dataset is large and license terms require
redistributing only your own derived code, not MVTec's images.

## Stage 2 — Classical ML baseline (anomaly detection)

`notebooks/02_baseline_anomaly_detection.ipynb` builds a baseline using
**One-Class SVM** and **Isolation Forest**, both trained only on normal
(`good`) images — faithful to MVTec AD's original anomaly-detection
protocol (see Stage 1's section 11 for why a naive supervised split would
have been wrong here). It:

- Loads the exact train/val/test split Stage 1 produced (or regenerates it
  with identical logic if run standalone).
- Extracts a 50-number hand-crafted feature vector per image (grid-based
  intensity/gradient statistics — see `src/features.py`).
- Scales features using statistics fit on train only.
- Trains both models on normal-only data, calibrates a decision threshold
  from validation scores only (never test labels), then evaluates once on
  the untouched test set (confusion matrix, precision/recall/F1, ROC-AUC).
- Saves baseline predictions to `data/processed/` for Stage 3 to compare
  against.

Run it the same way as Stage 1, just point Jupyter at
`notebooks/02_baseline_anomaly_detection.ipynb` instead.

## Stage 3 — Deep learning anomaly detection (PyTorch)

`notebooks/03_deep_learning_anomaly_detection.ipynb` builds two PyTorch
approaches, still following the same normal-only-training protocol:

- **Model A — custom convolutional autoencoder**, trained from scratch on
  normal images only, with a fully hand-written training loop. Anomaly
  score = per-image reconstruction error.
- **Model B — frozen, pretrained ResNet18** used as a feature extractor
  (genuine transfer learning, no fine-tuning), feeding the same One-Class
  SVM / Isolation Forest models from Stage 2, now with richer features.

Both are evaluated on the exact same untouched test set as Stage 2, and the
notebook produces a combined comparison table + ROC overlay across Stage 2
and both Stage 3 models.

**Requires internet access to download pretrained ImageNet weights** the
first time Model B runs (a few dozen MB, cached afterward at
`~/.cache/torch/hub/checkpoints/`). If that download fails for any reason,
the notebook automatically falls back to a randomly-initialized ResNet18
so the code still runs — but it prints a clear warning, and every output
computed that way is tagged `[RANDOM-INIT FALLBACK, NOT TRANSFER LEARNING]`
since it's no longer a real transfer-learning result at that point.

Run it the same way as Stage 1/2, pointing Jupyter at
`notebooks/03_deep_learning_anomaly_detection.ipynb`.

## Stage 4 — Defect localization (segmentation)

`notebooks/04_defect_localization.ipynb` extends both Stage 3 models to
produce a per-pixel anomaly **heatmap** instead of one score per image,
and evaluates it against MVTec's real pixel-level `ground_truth` masks —
still without ever training on those masks:

- **Model A** reuses Stage 3's saved autoencoder weights (training a quick
  fallback copy if they're not found) and keeps the full per-pixel
  reconstruction-error map instead of averaging it into one number.
- **Model B** extracts a spatial ResNet18 feature map (128 channels x 28 x
  28 patches instead of one pooled 512-dim vector) and measures each
  patch's distance to the average "normal" patch at that location seen
  during training — a simplified version of the published **PaDiM**
  technique.

Evaluation: IoU, Dice score, and strict pixel-level precision/recall/F1
(every pixel from every test image, including `good` ones). Bounding boxes
are drawn directly from each predicted mask's largest connected region —
a lightweight "detection" view with no separate detection model needed.

Same pretrained-weights-with-fallback behavior as Stage 3 applies to
Model B here too.

Run it the same way as the earlier notebooks, pointing Jupyter at
`notebooks/04_defect_localization.ipynb`.

## Stage 5 — Vision Transformers (CNN vs ViT)

`notebooks/05_vision_transformers.ipynb` adds a frozen, pretrained
**ViT-B/16** feature extractor — parallel to Stage 3's ResNet18 approach,
feeding the same One-Class SVM / Isolation Forest models — and measures
three things side by side for every backbone built so far (custom
autoencoder, ResNet18, ViT-B/16):

- **Accuracy** — ROC-AUC on the same untouched test set as every prior stage.
- **Model size** — parameter count and actual serialized weight size on disk.
- **Inference latency** — measured (not estimated) forward-pass time on
  your device.

We deliberately did **not** train a ViT from scratch — with ~177 training
images that would just demonstrate that ViTs need large pretraining
datasets by performing badly, which the notebook explains as a real,
worth-knowing limitation rather than something to accidentally rediscover.

Same pretrained-weights-with-fallback behavior as Stages 3-4 applies here.

Run it the same way as the earlier notebooks, pointing Jupyter at
`notebooks/05_vision_transformers.ipynb`. Expect it to run noticeably
slower than Stage 3/4 on CPU — ViT-B/16 was about 6x slower per batch than
ResNet18 in testing.

## Stage 6 — RAG + LLM

`notebooks/06_rag_llm.ipynb` builds a retrieval-augmented generation
pipeline over `data/knowledge_base/` — your own inspection SOP, defect
definitions guide, QC acceptance criteria, CAPA guide, and troubleshooting
guide — so the system can explain *why* a defect matters and what to do
next, grounded in real documentation rather than the vision model's bare
label.

**Pipeline:** Documents → chunking → local Sentence Transformer embeddings
(`all-MiniLM-L6-v2`, no API key/cost) → Chroma vector database → retrieve
top-k chunks → **Gemini API** (free tier) → grounded answer or refusal →
the same 30-question evaluation set as before.

**Requires one API key, set as an environment variable before starting
Jupyter:**

```bash
export GEMINI_API_KEY="your-key-here"
jupyter notebook notebooks/06_rag_llm.ipynb
```

(Windows PowerShell: `$env:GEMINI_API_KEY="your-key-here"`. Create a free
key at Google AI Studio.)

What it does:

- Parses the 5 documents into chunks at the boundaries they already
  declare (bold field headers in the defect guide; `**Section: X**`
  markers everywhere else) — verified chunk-by-chunk against the real
  files during development (60 chunks, all counts confirmed by hand).
- Embeds every chunk **locally** with Sentence Transformers (one-time
  model download the first time it runs, then fully offline — no
  per-call cost or network request) and stores them in a Chroma vector
  database with metadata (`document_id`, `defect_type`, `severity`).
- Retrieves relevant chunks per question, with optional `defect_type`
  filtering (for when a future vision-pipeline classification feeds
  directly into this step).
- Generates a grounded answer with Gemini, explicitly instructed to
  decline rather than guess when the retrieved context doesn't contain
  the answer.
- Runs your full 30-question evaluation set (`07_rag_test_questions.md`)
  automatically, reporting real retrieval accuracy and the refusal rate on
  the 9 questions specifically designed to be unanswerable.

**If the local embedding model or Gemini API is unavailable**, the
notebook falls back to a clearly-tagged offline substitute (TF-IDF for
embeddings — independently measured at ~95% retrieval accuracy on this
project's evaluation set — and a template stub for generation) so the
pipeline still runs for development. Generation failures are classified
specifically (missing key, invalid key, rate limit/quota, network error,
other) rather than crashing the notebook or silently mislabeling a
fallback as a real answer. Every fallback output is tagged in its printed
text.

## App — Streamlit web interface

`app.py` combines the CV pipeline (Stages 1-5) and the RAG system (Stage 6)
into a single interactive demo. It contains no new detection or retrieval
logic — it only wires together the same functions already built and
verified in `src/`.

**Honest scope note, shown in the app itself:** the CV pipeline is an
*anomaly detector* (Normal vs. Defective). It does not classify **which**
defect type is present — that classifier was never built or validated in
Stages 2-5. The app doesn't pretend otherwise: when a defect is flagged, it
asks the RAG system a generic "what should the operator do" question
rather than inventing a defect-type label.

### One-time setup: export trained model artifacts

The notebooks fit models fresh every run but never saved them — a live app
needs a persisted model to load instantly. Run this once (and again any
time you want to re-fit on updated data):

```bash
python scripts/export_inference_models.py
```

This reuses the exact same functions as Stages 3-4 (`build_resnet_feature_
extractor`, the same `OneClassSVM` hyperparameters, `calibrate_threshold`,
the same `ConvAutoencoder`) to fit and save:
- `models/resnet_ocsvm_bundle.joblib` — scaler + One-Class SVM + threshold
- `models/autoencoder_weights.pt` — reuses Stage 3's saved weights if
  present, otherwise trains a fresh copy
- `models/autoencoder_threshold.joblib` — Stage 4's per-pixel threshold

### Running the app

```bash
export GEMINI_API_KEY="your-key-here"
streamlit run app.py
```

**Image Inspection tab:** upload an image → runs the ResNet18+OCSVM
detector → shows Normal/Defective, anomaly score, threshold, and a
confidence heuristic (explicitly labeled as a heuristic, not a calibrated
probability) → runs the autoencoder's per-pixel heatmap for localization →
asks Gemini a grounded question about what to do next, with sources shown.

**Ask a Question tab:** free-text question, optional defect-type filter
(since the app can't auto-detect defect type, you can manually filter if
you already know it) → retrieves relevant chunks → grounded Gemini answer
with sources.

Both tabs handle Gemini failures gracefully and specifically (missing key,
invalid key, rate limit/quota, network error, other) — fallback responses
are always visibly labeled, never presented as if Gemini produced them.

### What I verified before shipping this

- The export script runs end-to-end and produces loadable artifacts.
- `predict_anomaly` and `compute_localization` were tested directly
  against a real image with real exported artifacts.
- The full app was run through Streamlit's `AppTest` framework (executes
  the actual script, not just a syntax check) for both tabs — the
  Ask-a-Question flow was driven through text input, button click,
  retrieval, and generation with zero exceptions.
- A full manual integration test replicated the Image Inspection tab's
  exact call sequence (detect → localize → retrieve → generate) against a
  real image end-to-end with zero exceptions.
- I could not test with a real `GEMINI_API_KEY` or real Sentence
  Transformer downloads in this sandbox (both blocked here, same
  restriction as every earlier stage) — both fallback paths were
  exercised and correctly labeled instead.

## API — FastAPI backend

`api.py` exposes the same CV and RAG capabilities as `app.py` through a
standalone HTTP service, so anything can call your models — not just the
Streamlit page. No new detection or retrieval logic here either: it wraps
the exact same `src/` functions, so the API and the Streamlit app will
always agree with each other. `app.py` is unchanged and still works
exactly as before — this is an additional way to reach the same models,
not a replacement.

### Running it

```bash
python scripts/export_inference_models.py   # if you haven't already
export GEMINI_API_KEY="your-key-here"
uvicorn api:app --reload
```

Interactive docs (try every endpoint from the browser): http://127.0.0.1:8000/docs

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Is everything loaded and ready? Reports detector/localization/RAG status, embeddings backend, device, whether a Gemini key is set. |
| `POST /predict` | Upload an image → `{label, score, threshold, confidence}`. Classification only. |
| `POST /detect` | Same as `/predict`, plus a bounding box and (optionally) the annotated image and heatmap as base64 PNGs. |
| `POST /ask` | `{question, defect_type?, top_k?}` → grounded answer + sources, with the same fallback labeling as the app. |
| `GET /metrics` | Request counts, error counts, and average latency per endpoint — in-memory, no external monitoring dependency. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@bottle.jpg"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is contamination?"}'
```

### What I verified before shipping this

- Every endpoint tested with FastAPI's `TestClient`, which triggers the
  real startup sequence (model loading, knowledge base embedding) rather
  than just checking the code parses.
- `/predict` against a real image returned results identical to `app.py`
  on the same image — confirming both share the exact same logic.
- `/detect`'s base64-encoded images were decoded back into real, correctly
  sized PNGs, not just checked for a non-empty string.
- `/ask` tested both with and without a `defect_type` filter, confirming
  the metadata filter narrows results correctly.
- Error handling tested directly: uploading a non-image file to `/predict`
  returns a proper `400` with a clear message instead of crashing.
- `/metrics` confirmed to accumulate real request/error/latency data
  across calls.


## Gemini setup (key + model)

1. Create a key in Google AI Studio. Copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=`.
   `.env` is gitignored and loaded automatically (python-dotenv), so it works in any terminal. Alternatively,
   set it in the SAME terminal that launches the app. In Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`.
   `export` only works on Linux/macOS, and `setx` only affects new terminals.
2. Gemini model names get retired (`gemini-2.0-flash` now returns HTTP 404). Run
   `python scripts/list_gemini_models.py` to see the models your key can use, then set `GEMINI_MODEL=<name>` in `.env`.
   The default is the alias `gemini-flash-latest`.


## Local LLM option: Ollama (no API key, no quota)

Answer generation can run fully on your machine through [Ollama](https://ollama.com), behind the same
`generate_answer()` call. Setup:

1. Install Ollama and start it (the desktop app runs it automatically; otherwise `ollama serve`).
2. `ollama pull llama3.2:3b` (about 2 GB; any chat model works, set `OLLAMA_MODEL` to use another).
3. Leave `GEMINI_API_KEY` unset, or force it with `LLM_PROVIDER=ollama` in `.env`. Restart the app.

`LLM_PROVIDER` is `auto` by default: Gemini if a key is set, otherwise Ollama if it is running with the model
pulled. The sidebar and `GET /health` (`llm_provider`) show which engine is active, and `/ask` returns the
`provider` that produced each answer. Fallback messages are labelled and are never presented as real answers.

Caveat: small local models follow the "refuse if not in the documents" rule less reliably than Gemini.
Re-run the 30-question evaluation with Ollama and compare the refusal rate before trusting it. I have not
measured it; the Ollama path was tested only against a simulated Ollama server, not a real model.


## Stage 9 — Edge AI optimization (ONNX + quantization)

Goal: make the vision backbone smaller and faster for a CPU-only edge device, and **measure** what that costs.

```bash
python scripts/export_inference_models.py   # once
python scripts/benchmark_onnx.py            # exports ONNX, quantizes, measures, writes reports/stage9_onnx_benchmark.md
```

It compares four variants of the ResNet18 feature extractor on your machine:
**PyTorch fp32, ONNX fp32, ONNX int8 (static, calibrated on 50 normal *train* images), ONNX int8 (dynamic)**.
For each it reports file size, CPU latency (mean/p50/p95, batch of 1), peak memory of a fresh process, feature
fidelity against PyTorch (cosine similarity, max difference), and the effect on the real task: ROC-AUC and
Normal/Defective agreement on the untouched MVTec test set (using the exported OCSVM head).

Concepts worth being able to explain:
- **ONNX** is a framework-neutral model format; ONNX Runtime executes it with optimized CPU kernels.
- **Static quantization** stores weights as int8 and needs *calibration* images to learn activation ranges.
  Calibration uses the normal train split only; using test images would leak test information.
- **Dynamic quantization** needs no calibration but is designed for Linear/MatMul layers, so it is often not a
  win for a convolutional network. It is in the comparison so that this is shown, not assumed.
- Whether int8 is *faster* depends on the CPU (int8 instructions such as VNNI). Size shrinks about 4x
  regardless. Latency and accuracy impact must be read from **your** benchmark output.

There are no benchmark numbers in this README on purpose. The only run I could do was a smoke test on synthetic
data with random weights. It confirmed the pipeline works, and showed int8 static being *slower* than ONNX fp32
on my sandbox CPU, which is exactly why you measure. Run the script on the real data and paste your table here.

## Stage 10 — Monitoring

What happens after deployment. Every prediction (API and Streamlit) is appended to `logs/predictions.jsonl`
with timestamp, **model version** (hash of the model file), label, score, confidence, latency and image
brightness/contrast. `src/monitoring.py` then provides:

- **Health summary:** counts, share of Defective predictions, confidence, latency p50/p95, per model version.
- **Drift detection** against a reference saved by the export script (normal validation images):
  *PSI* (how much the distribution moved: <0.1 stable, 0.1-0.25 moderate, >0.25 major) and the *KS test*
  (is the shift larger than chance). A signal is flagged only when both agree, KS limits are Bonferroni-corrected
  for testing three signals, PSI bins adapt to sample size (PSI is biased upward on small samples), and no
  verdict is given below 30 logged predictions or with a reference of fewer than 20 images.
- Surfaces: `GET /monitoring` (JSON) and the **Monitoring** tab in the app.

Measured by simulation (300-400 trials per cell, normal-distributed brightness, reference of 200 images; see
notebook 10): the false-alarm rate was about 4-5% at 30, 100 and 500 samples, and a brightness shift of one
standard deviation was detected in about 98% of trials even at 30 samples (a half-standard-deviation shift:
about 45% at 30 samples, 85% at 100, and essentially always at 500). That is a simulation of the statistics, not a
claim about your real camera.

Limits: no production labels means no accuracy tracking. A real deployment would also route a sample of bottles
to human review. Notebook `10_monitoring.ipynb` walks through the statistics.

## Stage 11 — Docker and deployment

```bash
python scripts/export_inference_models.py     # creates ./models (mounted into the containers)
docker compose up --build                     # UI :8501, API :8000, shared monitoring log
```

The image installs CPU-only PyTorch, bakes in the sentence-embedding model and ImageNet ResNet18 weights (so it
runs offline afterwards), and runs as a non-root user. Models are **not** in the image because they are fitted on
your data; they are mounted read-only. Put `GEMINI_API_KEY` in `.env` (read by compose) or run Ollama on the host
(compose points the containers at `host.docker.internal:11434`). See `DEPLOYMENT.md` for hosting options.

Honest status: I validated `docker-compose.yml` (`docker compose config`), ran the API under uvicorn, and ran the
exact health-check command against it. I could not build the image in my environment (no Docker daemon), so the
first `docker compose up --build` on your machine is the real test.

## Tests

`python -m pytest tests -q` covers: PSI/KS drift logic (no false alarm on the same distribution, drift on a
brightness shift, no verdict on tiny samples), the prediction log, LLM provider selection and fallbacks (Gemini
retry, daily quota, 404, Ollama present/absent, using a fake Ollama server), ONNX export fidelity and
quantization, and the API end to end (health, predict, monitoring, drift flagged, `/ask` fallback, bad upload).
The ONNX and API tests skip themselves when PyTorch or the exported models are missing, so CI stays light.
