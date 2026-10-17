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

UI design note: technical detail (raw model scores, thresholds, distances)
is real and accurate, but a factory-floor user shouldn't have to parse it
to get a straight answer. Every screen leads with a plain-language result;
technical numbers are one click away in an expander, never hidden or
fabricated-simpler -- just not front and center.

Run with: streamlit run app.py
Requires models/ to exist -- run `python scripts/export_inference_models.py` first.
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from inference import load_anomaly_model, predict_anomaly, load_autoencoder_model, compute_localization, draw_bounding_box
from rag_documents import load_knowledge_base
from rag_embeddings import build_embedder
from rag_generation import get_llm_client, llm_provider_name, generate_answer

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

DOC_TITLES = {
    "DOC-001": "Inspection Procedure (SOP)",
    "DOC-002": "Defect Definitions Guide",
    "DOC-003": "QC Acceptance Criteria",
    "DOC-004": "Corrective Action Guide",
    "DOC-005": "Troubleshooting Guide",
}

st.set_page_config(page_title="VisionAI Inspector", page_icon="🔍", layout="wide")

# NOTE: earlier versions of this app used custom-colored <div> HTML for
# result/guidance cards. That broke badly under Streamlit's dark theme --
# the hardcoded light background combined with the theme's light text
# color produced barely-readable text (confirmed from a real screenshot).
# Fixed by dropping custom-colored HTML entirely in favor of Streamlit's
# native st.container(border=True) + st.success/st.error/st.markdown,
# which automatically use readable, theme-correct colors in both light
# and dark mode -- this eliminates the whole bug class rather than
# patching one color value.


# ---------------------------------------------------------------- cached resource loading
@st.cache_resource(show_spinner="Loading models...")
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


@st.cache_resource(show_spinner="Preparing knowledge base (one-time)...")
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
    """Returns the active answer-generation client (Gemini or local Ollama); name kept for minimal diff."""
    return get_llm_client()


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
    "no_api_key": ("warning", "⚠️ No answer engine available. Either set GEMINI_API_KEY in your .env file, or install Ollama (ollama.com) and run `ollama pull llama3.2:3b`, then restart the app."),
    "ollama_not_running": ("warning", "⚠️ Ollama isn't running. Start the Ollama app (or run `ollama serve`) and try again."),
    "ollama_model_missing": ("error", "❌ The local model isn't downloaded yet. Run `ollama pull llama3.2:3b` (or the model set in OLLAMA_MODEL)."),
    "model_not_found": ("error", "❌ The configured Gemini model is no longer available. Run `python scripts/list_gemini_models.py`, then set GEMINI_MODEL in your .env file."),
    "invalid_api_key": ("error", "❌ The Gemini API key was rejected. Double check GEMINI_API_KEY."),
    "rate_limit_or_quota": ("warning", "⏳ Gemini's rate limit or free-tier quota was hit. Wait a bit and try again."),
    "server_error": ("warning", "⚠️ Gemini had a temporary server error — try again."),
    "network_error": ("warning", "⚠️ Couldn't reach Gemini — check your internet connection."),
    "empty_response": ("warning", "⚠️ Gemini didn't return an answer (possibly a safety filter)."),
    "other_error": ("error", "❌ Something went wrong calling Gemini."),
}


def render_guidance(result: dict):
    if result["used_fallback"]:
        kind, message = FALLBACK_MESSAGES.get(result["fallback_reason"], ("error", "❌ Generation unavailable."))
        getattr(st, kind)(message)
        with st.expander("Technical details (not a real answer)"):
            st.code(result["answer"])
        return

    if result.get("was_truncated"):
        st.warning("⚠️ This answer may have been cut short by a length limit.")

    with st.container(border=True):
        st.markdown(result["answer"])


def render_sources(retrieved: list[dict], friendly: bool = True):
    label = f"📚 Where this came from ({len(retrieved)} document excerpts)"
    with st.expander(label):
        for r in retrieved:
            doc_title = DOC_TITLES.get(r["document_id"], r["document_id"])
            st.markdown(f"**{r['document_id']} — {doc_title}**  ·  _{r['section']}_")
            snippet = r["text"][:280] + ("..." if len(r["text"]) > 280 else "")
            st.write(snippet)
            if not friendly:
                st.caption(f"distance: {r['distance']:.4f}")
            st.divider()


def draw_bbox(image: np.ndarray, bbox) -> Image.Image:
    return draw_bounding_box(image, bbox, color="red", width=3)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔍 VisionAI Inspector")
    st.caption("Industrial bottle inspection — CV + RAG demo")
    st.divider()

    device = get_device()
    anomaly_bundle, anomaly_error = get_anomaly_model()
    autoencoder, pixel_threshold, loc_error = get_localization_model()
    gemini_ready = bool(os.environ.get("GEMINI_API_KEY"))
    llm_client = get_gemini()
    provider = llm_provider_name(llm_client)

    st.markdown("**System status**")
    st.write(f"{'🟢' if anomaly_bundle is not None else '🔴'} Defect detector")
    st.write(f"{'🟢' if autoencoder is not None else '🔴'} Localization")
    provider_label = {"gemini": "Gemini (cloud)", "ollama": "Ollama (local)", "none": "not available"}[provider]
    st.write(f"{'🟢' if provider != 'none' else '🟡'} Guidance engine: {provider_label}")

    with st.expander("Technical status"):
        st.write(f"Device: `{device}`")
        if anomaly_bundle is not None:
            st.write("Detector backbone:", "pretrained ✅" if anomaly_bundle["resnet_pretrained"] else "random-init fallback ⚠️")
        st.write("GEMINI_API_KEY set:", gemini_ready)
        st.write("Active LLM provider:", provider)
        _, _, using_real_embeddings = get_rag_pipeline()
        st.write("Embeddings:", "Sentence Transformers ✅" if using_real_embeddings else "TF-IDF fallback ⚠️")

    st.divider()
    st.info(
        "**Scope note:** this system detects *Normal vs. Defective* only. "
        "It does **not** identify which specific defect is present — that "
        "wasn't built. Guidance shown is general inspection procedure, not "
        "defect-specific advice.",
        icon="ℹ️",
    )


# ---------------------------------------------------------------- main tabs
tab_inspect, tab_ask = st.tabs(["🖼️ Inspect an Image", "💬 Ask a Question"])

with tab_inspect:
    st.header("Inspect a bottle image")

    if anomaly_bundle is None:
        st.error("The defect detector isn't set up yet. Run `python scripts/export_inference_models.py` first.")
    else:
        uploaded_file = st.file_uploader("Upload a photo of a bottle", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            with st.spinner("Checking the image..."):
                result = predict_anomaly(image, anomaly_bundle, device)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, use_container_width=True, caption="Uploaded image")

            with col2:
                is_normal = result["label"] == "Normal"
                icon = "✅" if is_normal else "🚨"
                headline = "Looks Normal" if is_normal else "Defect Detected"
                subtext = (
                    "No anomaly found — this bottle looks consistent with a normal, defect-free unit."
                    if is_normal else
                    "This bottle looks different from a normal unit and has been flagged for review."
                )
                with st.container(border=True):
                    if is_normal:
                        st.success(f"**{icon} {headline}**")
                    else:
                        st.error(f"**{icon} {headline}**")
                    st.write(subtext)
                    st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.0%}")

                with st.expander("Technical details"):
                    st.write(f"Raw anomaly score: `{result['score']:.4f}`")
                    st.write(f"Decision threshold: `{result['threshold']:.4f}`")
                    st.caption(
                        "Score comes from a One-Class SVM trained only on normal images "
                        "(Stage 3). Confidence is a simple heuristic derived from distance "
                        "to the threshold, not a calibrated probability."
                    )

            if autoencoder is not None:
                st.subheader("Where's the issue?")
                loc = compute_localization(image, autoencoder, pixel_threshold, device)

                lcol1, lcol2 = st.columns([1, 1])
                with lcol1:
                    st.image(
                        draw_bbox(loc["resized_image"], loc["bounding_box"]),
                        use_container_width=True,
                        caption="Highlighted region the model flagged as unusual" if loc["bounding_box"] else "No specific region stood out",
                    )
                with lcol2:
                    with st.expander("Show technical heatmap"):
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.imshow(loc["heatmap"], cmap="inferno")
                        ax.axis("off")
                        st.pyplot(fig)
                        st.caption("Brighter = more different from what a normal bottle looks like at that pixel.")

            st.divider()
            st.subheader("What should I do?")

            if result["label"] == "Defective":
                query = ("A bottle was flagged as DEFECTIVE by the automated inspection system. "
                         "What should the operator do next according to inspection procedure?")
            else:
                query = ("A bottle was inspected and classified as NORMAL (no anomaly detected) "
                         "by the automated inspection system. Is any operator action needed?")

            collection, embed_query, using_real_embeddings = get_rag_pipeline()
            gemini_client = get_gemini()

            with st.spinner("Looking up guidance..."):
                retrieved = retrieve(collection, embed_query, query, top_k=TOP_K)
                gen_result = generate_answer(query, retrieved, client=gemini_client)

            render_guidance(gen_result)
            render_sources(retrieved)


with tab_ask:
    st.header("Ask a question")
    st.caption("Ask anything covered by the inspection documentation — procedures, defect definitions, acceptance criteria, and more.")

    with st.expander("Only if you already know the defect type (optional)"):
        defect_filter = st.selectbox(
            "Focus the search on:",
            options=[None, "contamination", "broken_small", "broken_large"],
            format_func=lambda x: "No filter — search everything" if x is None else x.replace("_", " ").title(),
        )

    question = st.text_input("Your question", placeholder="e.g. What counts as an acceptable amount of contamination?")

    if st.button("Ask", type="primary") and question.strip():
        collection, embed_query, using_real_embeddings = get_rag_pipeline()
        gemini_client = get_gemini()

        with st.spinner("Searching the documentation..."):
            retrieved = retrieve(collection, embed_query, question, top_k=TOP_K, defect_type=defect_filter)

        with st.spinner("Writing an answer..."):
            gen_result = generate_answer(question, retrieved, client=gemini_client)

        st.subheader("Answer")
        render_guidance(gen_result)
        render_sources(retrieved)
