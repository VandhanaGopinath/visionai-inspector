"""
api.py
======
FastAPI backend for the CV anomaly-detection pipeline (Stages 1-5) and the
RAG + Gemini system (Stage 6). This file contains NO new detection or
retrieval logic -- it wraps the exact same functions app.py uses, so both
the Streamlit UI and this API call identical underlying code and will
always agree with each other.

Why this exists alongside app.py rather than replacing it: app.py mixes
UI and model logic in one process, which means nothing else can use the
models except that one Streamlit page. This API exposes the same
capabilities as a standalone, headless service any client can call --
another script, a future non-Streamlit frontend, curl, a mobile app, an
automated test suite -- and is what makes this project independently
deployable (e.g. as a container) rather than tied to Streamlit's specific
hosting model.

app.py is UNCHANGED in how it works (still imports src/ functions
directly) -- this is an additional way to reach the same models, not a
replacement. Pointing app.py at this API instead is a reasonable future
step but wasn't asked for here, so it wasn't done.

Run with: uvicorn api:app --reload
Interactive docs at: http://127.0.0.1:8000/docs
Requires models/ to exist -- run `python scripts/export_inference_models.py` first.
"""

import os
import sys
import time
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import joblib
import chromadb
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from inference import (
    load_anomaly_model, predict_anomaly, load_autoencoder_model, compute_localization,
    draw_bounding_box, heatmap_to_image, encode_image_to_base64_png,
)
from rag_documents import load_knowledge_base
from rag_embeddings import build_embedder
from rag_generation import get_gemini_client, generate_answer

MODELS_DIR = PROJECT_ROOT / "models"
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
KNOWLEDGE_FILES = [
    "01_bottle_inspection_sop.md", "02_defect_definitions_guide.md", "03_qc_acceptance_criteria.md",
    "04_capa_guide.md", "05_troubleshooting_guide.md",
]
TOP_K = 4


# ---------------------------------------------------------------- lightweight in-process metrics
class Metrics:
    """
    Simple in-memory request counters and latency tracking -- intentionally
    minimal (no Prometheus/external dependency) since this is meant to
    demonstrate the concept (Stage 10's monitoring goal: know what's
    happening after deployment) without pulling in infrastructure this
    project doesn't otherwise need. Thread-safe for uvicorn's threaded
    request handling.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.total_latency_ms = defaultdict(float)

    def record(self, endpoint: str, latency_ms: float, is_error: bool = False):
        with self._lock:
            self.request_counts[endpoint] += 1
            self.total_latency_ms[endpoint] += latency_ms
            if is_error:
                self.error_counts[endpoint] += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = {
                ep: round(self.total_latency_ms[ep] / self.request_counts[ep], 2)
                for ep in self.request_counts
            }
            return {
                "request_counts": dict(self.request_counts),
                "error_counts": dict(self.error_counts),
                "average_latency_ms": avg_latency,
            }


metrics = Metrics()
state: dict = {}  # populated at startup: models, RAG pipeline, device


# ---------------------------------------------------------------- startup: load everything once
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["device"] = device

    try:
        state["anomaly_bundle"] = load_anomaly_model(MODELS_DIR, device)
    except FileNotFoundError as e:
        state["anomaly_bundle"] = None
        print(f"WARNING: anomaly model not loaded: {e}")

    try:
        state["autoencoder"] = load_autoencoder_model(MODELS_DIR / "autoencoder_weights.pt", device)
        ae_bundle = joblib.load(MODELS_DIR / "autoencoder_threshold.joblib")
        state["pixel_threshold"] = ae_bundle["pixel_threshold"]
    except FileNotFoundError as e:
        state["autoencoder"] = None
        state["pixel_threshold"] = None
        print(f"WARNING: localization model not loaded: {e}")

    chunks = load_knowledge_base(KB_DIR, filenames=KNOWLEDGE_FILES)
    embed_documents, embed_query, using_real_embeddings = build_embedder()
    chunk_texts = [c["text"] for c in chunks]
    chunk_vectors = embed_documents(chunk_texts)

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("bottle_inspection_kb")
    metadatas = [{
        "document_id": c["document_id"], "document_type": c["document_type"], "section": c["section"],
        "defect_type": c["defect_type"] or "general", "severity": c["severity"] or "unspecified",
    } for c in chunks]
    collection.add(ids=[c["chunk_id"] for c in chunks], embeddings=chunk_vectors.tolist(), metadatas=metadatas, documents=chunk_texts)

    state["collection"] = collection
    state["embed_query"] = embed_query
    state["using_real_embeddings"] = using_real_embeddings
    state["gemini_client"] = get_gemini_client()

    yield  # app runs here

    state.clear()


app = FastAPI(title="VisionAI Inspector API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _retrieve(query: str, top_k: int = TOP_K, defect_type: Optional[str] = None) -> list[dict]:
    query_vector = state["embed_query"](query)
    where = {"$or": [{"defect_type": defect_type}, {"defect_type": "general"}]} if defect_type else None
    results = state["collection"].query(query_embeddings=[np.asarray(query_vector).tolist()], n_results=top_k, where=where)
    retrieved = []
    for chunk_id, doc_text, meta, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({"chunk_id": chunk_id, "text": doc_text, "document_id": meta["document_id"],
                           "section": meta["section"], "distance": distance})
    return retrieved


# ---------------------------------------------------------------- request/response schemas
class BoundingBox(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class PredictResponse(BaseModel):
    label: str
    score: float
    threshold: float
    confidence: float


class DetectResponse(PredictResponse):
    bounding_box: Optional[BoundingBox]
    annotated_image_png_base64: Optional[str]
    heatmap_png_base64: Optional[str]


class AskRequest(BaseModel):
    question: str
    defect_type: Optional[str] = None
    top_k: int = TOP_K


class SourceChunk(BaseModel):
    document_id: str
    section: str
    text: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    used_fallback: bool
    fallback_reason: Optional[str]
    was_truncated: bool
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    device: str
    anomaly_detector_loaded: bool
    resnet_pretrained: Optional[bool]
    localization_model_loaded: bool
    rag_pipeline_ready: bool
    embeddings_backend: str
    gemini_api_key_set: bool


class MetricsResponse(BaseModel):
    request_counts: dict
    error_counts: dict
    average_latency_ms: dict


# ---------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    anomaly_bundle = state.get("anomaly_bundle")
    return HealthResponse(
        status="ok",
        device=str(state.get("device")),
        anomaly_detector_loaded=anomaly_bundle is not None,
        resnet_pretrained=anomaly_bundle["resnet_pretrained"] if anomaly_bundle else None,
        localization_model_loaded=state.get("autoencoder") is not None,
        rag_pipeline_ready=state.get("collection") is not None,
        embeddings_backend="sentence-transformers" if state.get("using_real_embeddings") else "tfidf-fallback",
        gemini_api_key_set=bool(os.environ.get("GEMINI_API_KEY")),
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    return MetricsResponse(**metrics.snapshot())


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Anomaly classification only (Normal/Defective + score) -- no localization."""
    start = time.perf_counter()
    if state.get("anomaly_bundle") is None:
        metrics.record("/predict", (time.perf_counter() - start) * 1000, is_error=True)
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded. Run scripts/export_inference_models.py first.")

    try:
        image = Image.open(io.BytesIO(await file.read()))
        result = predict_anomaly(image, state["anomaly_bundle"], state["device"])
        metrics.record("/predict", (time.perf_counter() - start) * 1000)
        return PredictResponse(**result)
    except Exception as e:
        metrics.record("/predict", (time.perf_counter() - start) * 1000, is_error=True)
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), include_images: bool = True):
    """Anomaly classification PLUS localization (bounding box, optionally annotated image + heatmap as base64 PNG)."""
    start = time.perf_counter()
    if state.get("anomaly_bundle") is None:
        metrics.record("/detect", (time.perf_counter() - start) * 1000, is_error=True)
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded. Run scripts/export_inference_models.py first.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        result = predict_anomaly(image, state["anomaly_bundle"], state["device"])

        bbox_model, annotated_b64, heatmap_b64 = None, None, None
        if state.get("autoencoder") is not None:
            loc = compute_localization(image, state["autoencoder"], state["pixel_threshold"], state["device"])
            if loc["bounding_box"] is not None:
                x0, y0, x1, y1 = loc["bounding_box"]
                bbox_model = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
            if include_images:
                annotated = draw_bounding_box(loc["resized_image"], loc["bounding_box"])
                annotated_b64 = encode_image_to_base64_png(annotated)
                heatmap_b64 = encode_image_to_base64_png(heatmap_to_image(loc["heatmap"]))

        metrics.record("/detect", (time.perf_counter() - start) * 1000)
        return DetectResponse(**result, bounding_box=bbox_model, annotated_image_png_base64=annotated_b64, heatmap_png_base64=heatmap_b64)
    except HTTPException:
        raise
    except Exception as e:
        metrics.record("/detect", (time.perf_counter() - start) * 1000, is_error=True)
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Retrieve relevant documentation and generate a grounded answer."""
    start = time.perf_counter()
    try:
        retrieved = _retrieve(request.question, top_k=request.top_k, defect_type=request.defect_type)
        result = generate_answer(request.question, retrieved, client=state.get("gemini_client"))
        metrics.record("/ask", (time.perf_counter() - start) * 1000, is_error=result["used_fallback"])
        return AskResponse(
            answer=result["answer"],
            used_fallback=result["used_fallback"],
            fallback_reason=result.get("fallback_reason"),
            was_truncated=result.get("was_truncated", False),
            sources=[SourceChunk(document_id=r["document_id"], section=r["section"], text=r["text"], distance=r["distance"]) for r in retrieved],
        )
    except Exception as e:
        metrics.record("/ask", (time.perf_counter() - start) * 1000, is_error=True)
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {e}")
