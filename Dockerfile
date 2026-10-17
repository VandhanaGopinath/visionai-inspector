# VisionAI Inspector -- one image, two services (API or UI), chosen by the run command.
# Models are NOT baked in (they are fitted on your data): mount ./models at /app/models.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache TORCH_HOME=/opt/torch-cache

WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel pulls ~GBs of CUDA libraries we don't need).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the pretrained weights into the image so the container works offline afterwards:
# the sentence-embedding model (RAG) and the ImageNet ResNet18 (defect detector backbone).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

COPY src ./src
COPY scripts ./scripts
COPY data/knowledge_base ./data/knowledge_base
COPY app.py api.py ./

RUN useradd --create-home appuser && mkdir -p /app/logs /app/models && chown -R appuser /app /opt/hf-cache /opt/torch-cache
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os,sys; p=os.environ.get('HEALTH_PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/health' if p=='8000' else f'http://localhost:{p}/_stcore/health', timeout=4).status==200 else 1)"

# Default: the REST API. docker-compose overrides this for the Streamlit UI.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
