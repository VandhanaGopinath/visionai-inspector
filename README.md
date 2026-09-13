# VisionAI Inspector — Stage 1: Data Analysis & Preprocessing

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
│   └── 06_rag_llm.ipynb
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
│   ├── rag_generation.py        # Gemini-based grounded generation (Stage 6)
│   └── inference.py             # single-image inference wrappers for the app (reuses Stage 3+4 logic)
├── scripts/
│   └── export_inference_models.py  # one-time: fits + saves models/ artifacts for the app
├── app.py                       # Streamlit web interface (CV + RAG combined)
├── requirements.txt
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
