# Deployment

What is verified and what is not: the code, API, app and tests were run locally. The Docker image build and
every hosting option below were **not** run by me; treat them as a checklist, not a guarantee.

## 1. Local (works today)

```bash
pip install -r requirements.txt
python scripts/export_inference_models.py
streamlit run app.py          # UI
uvicorn api:app --reload      # API (second terminal)
```

## 2. Docker on your machine or a server

```bash
python scripts/export_inference_models.py
docker compose up --build     # UI :8501, API :8000
```
- Models are mounted from `./models` (read-only); they are not baked into the image.
- Answers: put `GEMINI_API_KEY` in `.env`, or run Ollama on the host (containers reach it at `host.docker.internal:11434`).
- Monitoring log lives in the `logs` volume, shared by the UI and API.
- Update the model: re-run the export script, then `docker compose restart`. The model version hash changes and
  the drift reference is rebuilt with it.

## 3. A free public demo (options)

The honest constraint: the trained model files are fitted from the MVTec AD dataset, which is licensed
CC BY-NC-SA 4.0 (non-commercial). A portfolio demo is fine, but do not commit the dataset itself.

**Hugging Face Spaces (Docker SDK), free CPU tier** (suits the Streamlit UI only):
1. Create a Space with the Docker SDK and push this repository.
2. Space metadata (`README.md` front matter in the Space repo) needs `sdk: docker` and `app_port: 8501`.
3. The Dockerfile expects `./models` at runtime. Either upload the exported files (`models/*.joblib`,
   `models/autoencoder_weights.pt`, `models/monitoring_reference.json`) to the Space repo (check their size with
   `ls -lh models`, use Git LFS if large) and add `COPY models ./models`, or run the export in the build, which needs the dataset.
4. Set `GEMINI_API_KEY` as a Space **secret**, never in the repository. There is no Ollama on a free Space, so Gemini is the engine there.
5. Free Spaces sleep when idle, and the log file is not persistent across restarts.

**Render / Railway / Fly.io** work with the same Dockerfile; use the `ui` command from `docker-compose.yml` as
the start command and expose the port the platform provides. Free tiers have small RAM; the image loads PyTorch
and a sentence-embedding model, so check the memory limit first.

## 4. Before exposing it to others

- The API has open CORS and no authentication (`allow_origins=["*"]`). Add auth and restrict origins for anything public.
- Uploaded images are processed in memory and only brightness/contrast statistics are logged, not the images.
- Keep keys in environment variables or platform secrets. `.env` is gitignored and excluded from the image.
