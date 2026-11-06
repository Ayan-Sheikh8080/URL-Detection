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
| `GET /api/monitoring/stats` | Analysis totals, verdicts, latency, failures, and measured cloud cost |
| `GET /api/monitoring/drift` | PSI drift report against the training feature reference |
| `GET /metrics` | Prometheus metrics |
| `GET /api/screenshots/{filename}` | Retrieve a captured screenshot |

## Data, training, and tracking

The reproducible pipeline is in `dvc.yaml`. Place `PhiUSIIL_Phishing_URL_Dataset.csv` in `data/raw/`; preparation recognizes its `URL`/`label` schema, maps source label `0` (phishing) to the service's phishing label `1`, deduplicates identical URLs, and rejects conflicting labels. Dataset URLs containing user-info are accepted only in the offline training path; the API still rejects credential-bearing URLs.

```sh
python -m src.data.prepare --strict
python -m src.training.train
# or: dvc repro
```

It validates and normalizes the PhiUSIIL dataset and compares Logistic Regression, Random Forest, and XGBoost for URL-only, page-content-only, and combined models. Page models use the six fields available both in the dataset and live browser analysis: password-field presence, external-form presence, iframe count, redirect count, title length, and HTTPS. Reports include precision, recall, F1, confusion matrices, inference latency, and local cloud cost. MLflow tracks runs when installed; the full `requirements.txt` includes MLflow, DVC with S3 support, SHAP, and Evidently.

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
