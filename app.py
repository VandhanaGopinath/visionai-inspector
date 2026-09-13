"""
app.py
======
Streamlit interface combining the CV anomaly-detection pipeline (Stages
1-5) and the RAG + Gemini system (Stage 6). This file contains NO new
detection or retrieval logic -- it only wires together functions already
built and validated in src/:

- inference.py            (wraps Stage 3/4's ResNet+OCSVM detector and autoencoder localization)
- rag_documents.py        (Stage 6's document parsing)
- rag_embeddings.py       (Stage 6's Sentence Transformer embeddings)
- rag_generation.py       (Stage 6's Gemini generation)

Honest scope note, shown in the UI too: the CV pipeline is an ANOMALY
DETECTOR (Normal vs. Defective). It does not classify *which* defect type
an image shows -- that was never built or validated in Stages 2-5. So this
app does not claim a defect-type classification; it passes a generic
"flagged as defective" question to the RAG system rather than inventing a
type it can't actually determine.

Run with: streamlit run app.py
Requires models/ to exist -- run `python scripts/export_inference_models.py` first.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from inference import load_anomaly_model, predict_anomaly, load_autoencoder_model, compute_localization
from rag_documents import load_knowledge_base
from rag_embeddings import build_embedder
from rag_generation import get_gemini_client, generate_answer

MODELS_DIR = PROJECT_ROOT / "models"
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
KNOWLEDGE_FILES = [
    "01_bottle_inspection_sop.md",
    "02_defect_definitions_guide.md",
    "03_qc_acceptance_criteria.md",
    "04_capa_guide.md",
    "05_troubleshooting_guide.md",
]
TOP_K = 4

st.set_page_config(page_title="VisionAI Inspector", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------- cached resource loading
@st.cache_resource(show_spinner="Loading CV models...")
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading anomaly detector...")
def get_anomaly_model():
    device = get_device()
    try:
        return load_anomaly_model(MODELS_DIR, device), None
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_resource(show_spinner="Loading localization model...")
def get_localization_model():
    device = get_device()
    try:
        import joblib
        autoencoder = load_autoencoder_model(MODELS_DIR / "autoencoder_weights.pt", device)
        ae_bundle = joblib.load(MODELS_DIR / "autoencoder_threshold.joblib")
        return autoencoder, ae_bundle["pixel_threshold"], None
    except FileNotFoundError as e:
        return None, None, str(e)


@st.cache_resource(show_spinner="Building knowledge base (one-time)...")
def get_rag_pipeline():
    chunks = load_knowledge_base(KB_DIR, filenames=KNOWLEDGE_FILES)
    embed_documents, embed_query, using_real_embeddings = build_embedder()
    chunk_texts = [c["text"] for c in chunks]
    chunk_vectors = embed_documents(chunk_texts)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("bottle_inspection_kb")
    metadatas = [{
        "document_id": c["document_id"], "document_type": c["document_type"], "section": c["section"],
        "defect_type": c["defect_type"] or "general", "severity": c["severity"] or "unspecified",
    } for c in chunks]
    collection.add(ids=[c["chunk_id"] for c in chunks], embeddings=chunk_vectors.tolist(), metadatas=metadatas, documents=chunk_texts)

    return collection, embed_query, using_real_embeddings


@st.cache_resource(show_spinner=False)
def get_gemini():
    return get_gemini_client()


def retrieve(collection, embed_query, query: str, top_k: int = TOP_K, defect_type: str | None = None) -> list[dict]:
    query_vector = embed_query(query)
    where = {"$or": [{"defect_type": defect_type}, {"defect_type": "general"}]} if defect_type else None
    results = collection.query(query_embeddings=[np.asarray(query_vector).tolist()], n_results=top_k, where=where)
    retrieved = []
    for chunk_id, doc_text, meta, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({"chunk_id": chunk_id, "text": doc_text, "document_id": meta["document_id"],
                           "section": meta["section"], "distance": distance})
    return retrieved


FALLBACK_MESSAGES = {
    "no_api_key": ("warning", "⚠️ No GEMINI_API_KEY found. Set it as an environment variable and restart the app for real answers."),
    "invalid_api_key": ("error", "❌ Gemini rejected the API key as invalid. Check your GEMINI_API_KEY."),
    "rate_limit_or_quota": ("warning", "⏳ Gemini rate limit or free-tier quota exceeded. Try again shortly."),
    "server_error": ("warning", "⚠️ Gemini server error (likely transient) — try again."),
    "network_error": ("warning", "⚠️ Network error reaching Gemini — check your connection."),
    "empty_response": ("warning", "⚠️ Gemini returned an empty response (possibly a safety filter)."),
    "other_error": ("error", "❌ Unexpected error calling Gemini."),
}


def render_answer(result: dict):
    if result["used_fallback"]:
        kind, message = FALLBACK_MESSAGES.get(result["fallback_reason"], ("error", "❌ Generation unavailable."))
        getattr(st, kind)(message)
        with st.expander("Fallback details (not a real Gemini answer)"):
            st.code(result["answer"])
    else:
        st.markdown(result["answer"])


def render_sources(retrieved: list[dict]):
    with st.expander(f"📚 Sources ({len(retrieved)} chunks retrieved)"):
        for r in retrieved:
            st.markdown(f"**{r['document_id']} — {r['section']}**  _(distance: {r['distance']:.4f})_")
            st.caption(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))


def draw_bbox(image: np.ndarray, bbox) -> Image.Image:
    img = Image.fromarray(image).convert("RGB")
    if bbox is not None:
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline="red", width=2)
    return img


# ---------------------------------------------------------------- sidebar status
with st.sidebar:
    st.title("🔍 VisionAI Inspector")
    st.caption("Industrial bottle inspection — CV + RAG demo")
    st.divider()
    st.subheader("System status")

    device = get_device()
    st.write(f"**Device:** {device}")

    anomaly_bundle, anomaly_error = get_anomaly_model()
    if anomaly_bundle is not None:
        pretrained_note = "✅ pretrained" if anomaly_bundle["resnet_pretrained"] else "⚠️ random-init fallback"
        st.write(f"**Anomaly detector:** loaded ({pretrained_note})")
    else:
        st.error(f"**Anomaly detector:** not loaded\n\n{anomaly_error}")

    autoencoder, pixel_threshold, loc_error = get_localization_model()
    st.write(f"**Localization model:** {'loaded' if autoencoder is not None else 'not loaded'}")

    st.write(f"**GEMINI_API_KEY set:** {bool(os.environ.get('GEMINI_API_KEY'))}")

    st.divider()
    st.caption(
        "Scope note: this system detects Normal vs. Defective. It does NOT classify "
        "*which* defect type is present — that classifier was never built. Guidance "
        "below is general inspection procedure, not defect-specific advice."
    )


# ---------------------------------------------------------------- main tabs
tab_inspect, tab_ask = st.tabs(["🖼️ Image Inspection", "💬 Ask a Question"])

with tab_inspect:
    st.header("Image Inspection")

    if anomaly_bundle is None:
        st.error("Anomaly detector isn't available. Run `python scripts/export_inference_models.py` first.")
    else:
        uploaded_file = st.file_uploader("Upload a bottle image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded image")
                st.image(image, use_container_width=True)

            with st.spinner("Running anomaly detection..."):
                result = predict_anomaly(image, anomaly_bundle, device)

            with col2:
                st.subheader("Detection result")
                if result["label"] == "Normal":
                    st.success(f"✅ **{result['label']}**")
                else:
                    st.error(f"🚨 **{result['label']}**")
                st.metric("Anomaly score", f"{result['score']:.4f}", help="Raw One-Class SVM decision_function value. Lower = more anomalous.")
                st.metric("Decision threshold", f"{result['threshold']:.4f}")
                st.progress(result["confidence"], text=f"Confidence heuristic: {result['confidence']:.1%} (not a calibrated probability)")

            if autoencoder is not None:
                st.subheader("Localization (where the anomaly appears to be)")
                loc = compute_localization(image, autoencoder, pixel_threshold, device)

                lcol1, lcol2, lcol3 = st.columns(3)
                with lcol1:
                    st.caption("Resized input")
                    st.image(loc["resized_image"], use_container_width=True)
                with lcol2:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(loc["heatmap"], cmap="inferno")
                    ax.axis("off")
                    st.caption("Reconstruction-error heatmap")
                    st.pyplot(fig)
                with lcol3:
                    st.caption("Predicted anomalous region")
                    st.image(draw_bbox(loc["resized_image"], loc["bounding_box"]), use_container_width=True)

            st.divider()
            st.subheader("Inspection guidance")

            if result["label"] == "Defective":
                query = ("A bottle was flagged as DEFECTIVE by the automated inspection system. "
                         "What should the operator do next according to inspection procedure?")
            else:
                query = ("A bottle was inspected and classified as NORMAL (no anomaly detected) "
                         "by the automated inspection system. Is any operator action needed?")

            collection, embed_query, using_real_embeddings = get_rag_pipeline()
            gemini_client = get_gemini()

            if not using_real_embeddings:
                st.info("ℹ️ Using TF-IDF fallback embeddings (Sentence Transformer model unavailable).")

            with st.spinner("Retrieving guidance and generating explanation..."):
                retrieved = retrieve(collection, embed_query, query, top_k=TOP_K)
                gen_result = generate_answer(query, retrieved, client=gemini_client)

            render_answer(gen_result)
            render_sources(retrieved)


with tab_ask:
    st.header("Ask a Question")
    st.caption("Ask anything covered by the inspection documentation (SOP, defect definitions, QC criteria, CAPA, troubleshooting).")

    defect_filter = st.selectbox(
        "Filter by defect type (optional — only if you already know it)",
        options=[None, "contamination", "broken_small", "broken_large"],
        format_func=lambda x: "No filter" if x is None else x,
    )

    question = st.text_input("Your question", placeholder="e.g. What is the acceptance criteria for contamination?")

    if st.button("Ask", type="primary") and question.strip():
        collection, embed_query, using_real_embeddings = get_rag_pipeline()
        gemini_client = get_gemini()

        if not using_real_embeddings:
            st.info("ℹ️ Using TF-IDF fallback embeddings (Sentence Transformer model unavailable).")

        with st.spinner("Retrieving relevant documentation..."):
            retrieved = retrieve(collection, embed_query, question, top_k=TOP_K, defect_type=defect_filter)

        with st.spinner("Generating answer..."):
            gen_result = generate_answer(question, retrieved, client=gemini_client)

        st.subheader("Answer")
        render_answer(gen_result)
        render_sources(retrieved)
