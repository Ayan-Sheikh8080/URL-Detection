# Phishing URL Analyzer

FastAPI service that combines URL-model predictions with read-only Chromium evidence. The browser follows redirects, records the final domain, visible page text, forms, iframes, TLS status, and a screenshot. The API returns a `safe`, `suspicious`, or `phishing` verdict with contributing factors.

## Run locally

Use Python 3.12 for the complete optional MLOps stack. On Windows, run `.\start.ps1` from PowerShell; it installs dependencies, initializes local DVC metadata, runs the data/training stages, installs Chromium, and starts the API. To run the app with only the runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.\.venv\Scripts\python.exe -m playwright install chromium
npm ci
npm start
```

`npm start` uses `.venv` when it exists. If trained artifacts are missing, it prepares the dataset and trains the models before starting the API on port `8000`.

The full stack can also be started in isolated containers:

```sh
docker compose up --build
```

Compose starts the API, a separate browser sandbox with no AWS credentials, and MLflow. The browser container drops Linux capabilities, runs as a non-root user, disables downloads, and shares only the screenshot volume with the API. Direct local `npm start` is a development mode: browser work runs in the API process, so use Docker Compose for the isolated-browser deployment.

## API

Interactive docs: `http://localhost:8000/docs`

```sh
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","model_config":"combined"}'
```

`model_config` accepts `url_only`, `content_only`, or `combined`; `browser_enabled` can disable browsing for a request. `use_llm: true` opts into Amazon Bedrock. The browser never submits forms or downloads files, and rejects non-HTTP(S), credential-bearing, local, private, and non-publicly-resolved targets.

Other endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service and model readiness |
| `GET /api/models/versions` | Version-tagged local model registry manifest |
| `GET /api/monitoring/stats` | Analysis totals, verdicts, latency, failures, and measured cloud cost |
| `GET /api/monitoring/drift` | PSI drift report against the training feature reference |
| `GET /metrics` | Prometheus metrics |
| `GET /api/screenshots/{filename}` | Retrieve a captured screenshot |

## Data, training, and tracking

The reproducible pipeline is in `dvc.yaml`. The source CSV remains Git-versioned so clean checkouts and CI can reproduce without an unconfigured cloud remote; the first DVC stage creates a hash-verified snapshot of that raw dataset, and DVC versions the snapshot, processed dataset, trained models, and retrieval index. Configure a DVC remote to share the DVC-managed artifacts between machines. Preparation recognizes the PhiUSIIL `URL`/`label` schema, maps source label `0` (phishing) to the service's phishing label `1`, deduplicates identical URLs, and rejects conflicting labels. Dataset URLs containing user-info are accepted only in offline training; the RAG corpus removes URL user-info and query values, and the API rejects credential-bearing URLs.

```sh
python -m src.data.prepare --strict
python -m src.training.train
# or reproduce prepare, train, and rag_index together:
dvc repro
```

It validates and normalizes the PhiUSIIL dataset and compares Logistic Regression, Random Forest, and XGBoost for URL-only, page-content-only, and combined models. Page models use the six fields available both in the dataset and live browser analysis: password-field presence, external-form presence, iframe count, redirect count, title length, and HTTPS. Reports include precision, recall, F1, confusion matrices, inference latency, and local cloud cost. Training uses CPU-only XGBoost, logs candidate model artifacts and metrics to MLflow, and writes version-tagged model bundles under `models/registry/`. The full `requirements.txt` includes MLflow, DVC with S3 support, SHAP, and Evidently.

The RAG stage ingests `data/rag_source/*.md` and a deterministic, label-balanced sample of the processed dataset, chunks documents, generates 2,048-dimensional normalized hashing-vector embeddings, and persists them in a SQLite cosine-similarity vector index. This offline lexical retriever has no external service or embedding-model download requirement; retrieved cases are supporting context, not ground truth. Add `.md` or `.txt` references under `data/rag_source/`, then run `dvc repro rag_index`. The API returns its retrieved context and bounded specialist-agent trace with each analysis. Run the offline agent harness against a small sample with `python -m src.agent.harness --limit 25`.

An optional Hugging Face fine-tuning pipeline preprocesses balanced URL/page-feature text, tokenizes it with a pretrained tokenizer, trains DistilBERT on CPU, and reports a stratified validation set. Its DVC stages are kept separate from the default pipeline so ordinary runs do not download PyTorch or model weights:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements-hf.txt
dvc repro pipelines\huggingface\dvc.yaml
```

The Hugging Face model, tokenizer, dataset hash, validation metrics, and version tags are saved under `models/huggingface/`; the dedicated DVC pipeline tracks the sampled dataset, model artifacts, and validation metrics, and MLflow logs the model artifacts and metrics when available. Model and tokenizer downloads require access to the Hugging Face Hub.

To use an S3 DVC remote, configure a bucket and AWS credentials outside the repository:

```sh
dvc remote add --default storage s3://YOUR_BUCKET/YOUR_PREFIX
dvc push
```

Never commit AWS keys. Configure `MLFLOW_TRACKING_URI` to use a remote MLflow server. Bedrock uses the API process's AWS identity; the browser container does not receive it. A Bedrock call's cost is reported as unknown unless billing data is separately configured.

## CI and AWS deployment

GitHub Actions validates data, trains and tests the models, runs Python and Node tests/lint, and builds both container images. AWS deployment is opt-in: set the repository variable `ENABLE_AWS_DEPLOY=true`, configure the `AWS_ROLE_TO_ASSUME` secret, and set repository variables `AWS_REGION`, `API_ECR_REPOSITORY`, `SANDBOX_ECR_REPOSITORY`, `ECS_CLUSTER`, and `ECS_SERVICE`. The ECS service/task definition and networking must already exist and reference the pushed `latest` images; the API and sandbox must share screenshot storage and the API must use `http://localhost:8080` for the sandbox.

## Dataset limitation

The PhiUSIIL dataset is a large but static benchmark. Its holdout results do not establish production performance or generalization to current phishing campaigns. Validate on a separate, time-based dataset before deployment.
