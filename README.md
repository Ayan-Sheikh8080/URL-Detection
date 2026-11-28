# PhishGuard — Phishing URL Analyzer

PhishGuard is a local-first phishing URL analysis project. It combines trained URL and page-content models with optional browser evidence, semantic retrieval (RAG), and optional Amazon Bedrock reasoning. A React dashboard provides URL scanning, results, recent session scans, and API monitoring.

> **Use responsibly:** Analyze only URLs you are authorized to inspect. The analyzer rejects non-HTTP(S), credential-bearing, local, private, and non-publicly-resolved targets. Browser inspection is read-only: it does not submit forms or download files.

## What it does

- Classifies URLs as `safe`, `suspicious`, or `phishing`.
- Supports `url_only`, `content_only`, and `combined` model configurations.
- Collects browser evidence such as redirects, page title and text, forms, iframes, TLS status, and an optional screenshot.
- Uses a local semantic index to retrieve supporting references; retrieved content is evidence, not ground truth.
- Optionally uses Amazon Bedrock for additional reasoning. The model-based result remains available without Bedrock.
- Shows the verdict, confidence, model scores, contributing reasons, browser evidence, RAG results, and agent trace in the dashboard.
- Exposes health, model-version, monitoring, drift, screenshot, and Prometheus endpoints.

The dashboard's **Recent Scans** list and notifications are kept in the current browser session. Aggregate monitoring counts are provided by the API and persisted in its monitoring event file (or Docker volume).

## Architecture

| Component | Purpose |
|---|---|
| `frontend/` | React + Vite dashboard; proxies `/api` requests to `http://localhost:8000` during development |
| `src/api/` | FastAPI analysis and monitoring endpoints |
| `src/ml/`, `src/training/` | Feature extraction, model training, and model registry |
| `src/agent/` | Browser evidence collection and evidence-weighted reasoning |
| `src/rag/` | Local semantic retrieval index and query pipeline |
| `src/monitoring/` | Durable analysis events and drift metrics |
| `Dockerfile.sandbox` | Optional isolated Playwright browser service |
| `dvc.yaml` | Reproducible dataset, model, and RAG-index pipeline |

## Requirements

- Python 3.12 (used by CI and recommended for local development)
- Node.js 20 or later and npm (frontend and optional root npm scripts)
- Git
- Chromium for local browser analysis; installed by the Playwright command below
- Docker Desktop with Compose, only if using the containerized stack

The first local startup may take a while: if model artifacts or the RAG index are missing, the API prepares the dataset, trains the models, and builds the index. FastEmbed may download its embedding model on first use.

## Run locally

Run the API and dashboard in separate terminals from the repository root.

### 1. Set up and start the API

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
python -m playwright install chromium
python -m src.start
```

If PowerShell blocks activation, allow it for the current terminal only, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Git Bash on Windows**

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
python -m playwright install chromium
python -m src.start
```

The API listens on port `8000`. Its interactive documentation is at `http://localhost:8000/docs`; check readiness at `http://localhost:8000/api/health`.

For development with auto-reload, first make sure model and RAG artifacts exist, then use this instead of `python -m src.start`:

```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Install and start the dashboard

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The Vite development server proxies API requests to port `8000`. If the API is already running—for example, from Docker Compose—use that API instead of starting another server on the same port.

If PowerShell reports that `npm.ps1` cannot run because of the execution policy, use `npm.cmd` in place of `npm` (for example, `npm.cmd run dev`).

### Optional root npm launcher

The root `npm start` script is an alternative API launcher. Install root Node dependencies with `npm ci`, then run `npm start`; it uses `.venv` when available and starts `src.start`. The React dashboard has its own dependencies and scripts under `frontend/`.

## Run with Docker Compose

Docker Compose starts the API, isolated browser sandbox, MLflow, and Prometheus. The dashboard is still started separately using the frontend commands above.

The Compose file mounts the host AWS configuration directory into the API container. Set `AWS_CONFIG_DIR` to a directory that exists on your machine before starting Compose; Bedrock also needs valid AWS credentials when enabled.

**PowerShell**

```powershell
$env:AWS_CONFIG_DIR = "$HOME/.aws"
docker compose up --build
```

**Git Bash**

```bash
export AWS_CONFIG_DIR="$HOME/.aws"
docker compose up --build
```

Stop the containers with `docker compose down`. Compose exposes the API on `http://localhost:8000`, the browser sandbox on `http://localhost:8080`, MLflow on `http://localhost:5000`, and Prometheus on `http://localhost:9090`.

For local development without Compose, the API runs browser analysis in-process. Use Compose when you need the separate browser container.

## API

### Analyze a URL

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","model_config":"combined","use_llm":false,"browser_enabled":true}'
```

`model_config` accepts `url_only`, `content_only`, or `combined`. `browser_enabled` controls browser evidence collection. Set `use_llm` to `true` to request Amazon Bedrock reasoning; this requires AWS credentials and may incur cloud charges.

### Endpoints

| Method and path | Purpose |
|---|---|
| `GET /` | API summary and endpoint links |
| `GET /api/health` | Service and model readiness |
| `GET /api/healthz` | Health-check alias |
| `GET /api/models/versions` | Registered model versions |
| `POST /api/analyze` | Analyze a URL |
| `GET /api/monitoring/stats` | Analysis totals, verdict counts, latency, browser failures, and cost measurements |
| `GET /api/monitoring/drift` | Feature-drift report |
| `GET /api/screenshots/{filename}` | Retrieve a captured screenshot |
| `GET /metrics` | Prometheus metrics |

## Data, training, and RAG

The source dataset is `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`. Preparation validates and normalizes it, removes duplicate URLs, and rejects conflicting labels. For the PhiUSIIL schema, source label `0` (phishing) is mapped to the service's phishing label `1`. The dataset is large and static; benchmark results do not establish performance on current phishing campaigns.

The DVC pipeline in `dvc.yaml` snapshots the raw CSV, prepares the dataset, trains models, and builds the retrieval index:

```bash
python -m pip install -r requirements.txt
python -m dvc repro
```

The training pipeline compares Logistic Regression, Random Forest, and CPU-only XGBoost for URL, page-content, and combined models. Reports include precision, recall, F1, confusion matrices, inference latency, and local cost estimates. Versioned model bundles are written under `models/registry/`; training metrics are written under `reports/`.

The page model uses fields available from the source dataset and live browser analysis: password-field presence, external-form presence, iframe count, redirect count, title length, and HTTPS. The RAG stage indexes `data/rag_source/analysis_guide.md` and a deterministic, label-balanced sample of processed data using FastEmbed (`sentence-transformers/all-MiniLM-L6-v2`) and a local SQLite vector store.

To run stages individually:

```bash
python -m src.data.prepare --strict
python -m src.training.train
python -m src.rag.pipeline build
```

Add trusted `.md` or `.txt` references under `data/rag_source/` and rebuild the index with `python -m src.rag.pipeline build` (or `python -m dvc repro rag_index`). To run the offline agent harness on a small sample:

```bash
python -m src.agent.harness --limit 25
```

An optional Hugging Face fine-tuning pipeline is separated from the default DVC pipeline so it does not download PyTorch or transformer weights during ordinary runs:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements-hf.txt
dvc repro pipelines\huggingface\dvc.yaml
```

The optional model and tokenizer are saved under `models/huggingface/`; this workflow requires access to the Hugging Face Hub.

To share DVC-managed artifacts, configure a remote and credentials outside the repository, then use `dvc push`. Never commit AWS keys, tokens, or credential files.

## Tests and checks

Install the full Python development dependencies and root Node dependencies:

```bash
python -m pip install -r requirements-dev.txt
npm ci
```

Run the Python and root JavaScript tests and lint:

```bash
python -m pytest -q
npm test -- --runInBand
npm run lint
```

Run the frontend lint and production build from `frontend/`:

```bash
cd frontend
npm ci
npm run lint
npm run build
```

## CI and AWS deployment

GitHub Actions runs on pushes to `main` and `develop`, pull requests to `main`, and manual dispatch. The workflow tests the Python and root Node projects and builds the API and sandbox images. The frontend has its own lint and build commands described above.

The workflow's AWS deployment job is opt-in and runs only for `main` when the `ENABLE_AWS_DEPLOY` repository variable is set to `true`. Configure the AWS credentials and repository variables expected by `.github/workflows/ci-cd.yml` before enabling deployment. Review `ecs-task-definition.json` and `github-deploy-policy.json` for environment-specific account IDs, role ARNs, regions, repository names, and permissions before using or publishing them.

## Limitations

- A verdict is a risk signal, not a guarantee. Review the reasons and evidence before acting.
- The benchmark dataset is static and may not represent new phishing campaigns. Evaluate on separate, time-based data before deployment.
- The dashboard's recent scan list is browser-session-only; it is not a server-side per-user history.
- Bedrock and Hugging Face downloads require external network access; Bedrock requires AWS credentials and can incur charges.
