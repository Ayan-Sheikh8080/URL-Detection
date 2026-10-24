# Agentic Phishing Link Analysis

A sandboxed-browser agent with an end-to-end MLOps pipeline.

**Team:** Ayan Haseeb (22L-7210) · Hamza Asghar (22L-7792) · Izza Javed (21L-7990)

---

## 1. What this project does

URL-only classifiers judge a link from its text. They cannot see what the page
actually *does* — so they miss fresh domains, redirect chains and lookalike
pages. This project builds an agent that:

1. scores the URL with a trained ML model;
2. visits the link in an **isolated, sandboxed headless browser**;
3. collects live evidence: redirect chain, final domain, visible text, forms
   (password fields, cross-domain submission targets), iframes and the TLS
   certificate;
4. compares the final domain against the original link and the brand shown on
   the page;
5. weighs everything in a reasoning layer and returns a verdict
   (`safe` / `suspicious` / `phishing`) with confidence and plain-language
   reasons.

All of this is delivered as a reproducible pipeline: versioned data, tracked
experiments, a containerized API, automated tests, and monitoring.

---

## 2. Architecture

```
                       ┌──────────────────────────────────────────┐
   POST /analyze       │              Agent (API)                 │
   {"url": "..."} ───► │  PhishingAgent.analyze()                 │
                       │                                          │
                       │  ┌────────────────┐  tool 1: URL model   │
                       │  │ ModelRegistry  │◄── models/*.pkl       │
                       │  └────────────────┘                      │
                       │  ┌────────────────┐  tool 2: browse      │
                       │  │ SandboxClient  │──────────┐            │
                       │  └────────────────┘          │            │
                       │  ┌────────────────┐  tool 3: cert/domain │
                       │  │ checks.py      │◄── TLS + brand rules │
                       │  └────────────────┘                      │
                       │  ┌────────────────┐  tool 4: reasoning   │
                       │  │ Reasoner       │  rule-based | Bedrock │
                       │  └────────────────┘                      │
                       └───────────────────┬──────────────────────┘
                                           │ HTTP /visit
                       ┌───────────────────▼──────────────────────┐
                       │   Browser sandbox (own container)        │
                       │   Playwright + headless Chromium         │
                       │   no credentials · no cloud role         │
                       │   observe-only · timeouts · no downloads │
                       │   private/metadata networks blocked      │
                       └──────────────────────────────────────────┘

   MLOps around it:
   data/raw ──DVC──► prepare ──► train (MLflow: 3 configs x 3 families) ──► evaluate
                                        │
                                        ├─► models/index.json ──► API
                                        └─► reference_features.csv ──► drift monitoring
```

**Safety posture.** The browser runs only in the sandbox container, which has no
credentials and no cloud role, blocks `file:`/`data:`/`javascript:` schemes and
private/metadata addresses, runs observe-only (no clicks, no form submission, no
downloads) under a hard wall-clock timeout. Page text is treated as **untrusted
data**: it is quarantined in the prompt, the system prompt forbids following
instructions found inside it, and the ML score is supplied as an independent
second opinion the LLM cannot override.

---

## 3. Repository layout

```
src/
  config.py                 runtime settings (env-driven)
  features/
    url_features.py         lexical URL features (27 features)
    page_features.py        page-evidence features (18 features)
    featurize.py            URL / content / combined matrices
  agent/
    agent.py                orchestration -> AnalysisResult
    browser_sandbox.py      sandboxed Playwright visit + safety policy
    sandbox_client.py       remote sandbox client (local fallback)
    evidence.py             evidence dataclasses + summary
    checks.py               TLS certificate + domain/brand checks
    reasoning.py            rule-based and Bedrock reasoners, prompt hardening
  data/
    validate.py             missing / duplicate / invalid data checks
    prepare.py              validate -> dedupe -> stratified split
  training/
    common.py               model factories, metrics, matrices
    train.py                3 configs x 3 families, MLflow tracking
    evaluate.py             comparison report + latency
  ml/model_registry.py      promoted models, scoring, factor attribution
  serving/
    api.py                  FastAPI: /analyze, /healthz, /monitoring/*
    schemas.py              request/response models
  monitoring/
    metrics.py              verdict / latency / browser-failure stats
    drift.py                PSI (Evidently optional) drift report
data/raw/                   sample_urls.csv, crawled_evidence.jsonl
dvc.yaml params.yaml        reproducible pipeline
Dockerfile.api              agent + API image
Dockerfile.sandbox          Playwright sandbox image
docker-compose.yml          local end-to-end stack
.github/workflows/ci.yml    tests, data validation, safety, builds, ECR push
```

---

## 4. Quickstart

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 1) Prepare data (validates, dedupes, splits)
python -m src.data.prepare            # or: dvc repro prepare

# 2) Train and compare all configurations (tracked in MLflow)
python -m src.training.train          # or: dvc repro
python -m src.training.evaluate

# 3) Run the API (local in-process browser, dev only)
uvicorn src.serving.api:app --port 8000

# 4) Or run the full containerized stack (API + sandbox)
docker compose up --build
```

Analyze a link:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://paypal-secure-login.com/signin"}'
```

```json
{
  "url": "https://paypal-secure-login.com/signin",
  "verdict": "phishing",
  "confidence": 0.92,
  "reasons": [
    "The link redirected the visitor to a different domain (account-secure-verify.top).",
    "The page contains a password field whose form submits to a different domain than the page itself.",
    "The page presents itself as \"paypal\" but is hosted on account-secure-verify.top."
  ],
  "model_score": 0.88,
  "combined_score": 0.95,
  "top_factors": [{"feature": "suspicious_keyword_count", "contribution": 0.41, "direction": "increases_risk"}],
  "evidence_summary": {"final_url": "https://account-secure-verify.top/login", "redirect_count": 2},
  "browser": {"ok": true, "load_ms": 1420.5},
  "reasoner": "rule_based"
}
```

Other endpoints: `GET /healthz`, `GET /monitoring/stats`, `GET /monitoring/drift`.

---

## 5. Data

| Dataset | Purpose | Notes |
|---|---|---|
| `data/raw/sample_urls.csv` | bundled synthetic sample (80 URLs, balanced) | lets the whole pipeline run out of the box |
| PhiUSIIL Phishing URL dataset | real URL-only training | download the CSV, point `data.raw_path` at it and set `data.label_positive_is: legitimate` (PhiUSIIL labels 1 = legitimate) |
| `data/raw/crawled_evidence.jsonl` | content-only and combined training | `{url, label, evidence}` records produced by the sandbox; a small sample is included |

Validation (`src/data/validate.py`) fails the pipeline on missing values,
invalid URLs, labels outside `{0,1}`, single-class data and missing columns; it
warns on duplicates and heavy class imbalance. `prepare` dedupes, drops
invalids and writes a stratified split plus `dataset_meta.json` (row counts,
class share, source SHA-256).

Datasets and preprocessing outputs are DVC-tracked (`dvc.yaml`). Configure an S3
remote once:

```bash
dvc remote add -d storage s3://my-phishing-dvc-remote/dvc
dvc push
```

---

## 6. Models and evaluation

Three families — **Logistic Regression**, **Random Forest**, **XGBoost** — are
trained in three configurations:

| Config | Features | Training data |
|---|---|---|
| `url_only` | 27 lexical URL features | full URL dataset |
| `content_only` | 18 page-evidence features | crawled evidence |
| `combined` (agent) | URL + page features | crawled evidence |

Every run is logged to MLflow (`phishing-link-detection`): hyper-parameters,
precision, recall, F1, ROC-AUC, confusion matrix, feature schema and the model
artifact. The best model per configuration is promoted to
`models/<config>_model.pkl` with an `index.json` manifest; `reference_features.csv`
captures the training distribution for drift monitoring.

`python -m src.training.evaluate` writes `reports/evaluation.json`, including
per-configuration precision/recall/F1, confusion matrices, per-sample inference
latency, and the **relative F1 improvement of the agent over URL-only** — the
headline number the proposal asks for.

> **Reading the sample-data numbers.** The bundled dataset is synthetic and the
> phishing URLs are lexically obvious, so `url_only` reaches F1 = 1.0 on it —
> that is a pipeline check, not an accuracy claim. The synthetic evidence
> records are also few, so `content_only` is the weakest configuration here.
> Run against PhiUSIIL plus a real crawled set for meaningful comparison
> numbers; the pipeline and reports are identical either way.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect experiments
```

---

## 7. Reasoning layer

`LLM_PROVIDER=rule_based` (default) uses a transparent weighted-evidence
reasoner — deterministic, offline, and what the tests exercise.
`LLM_PROVIDER=bedrock` calls an LLM with a hardened prompt; if Bedrock or
credentials are unavailable it silently falls back to the rule-based reasoner so
analysis never fails.

Prompt-injection defence, verified by tests in
`tests/test_safety_and_injection.py`:

* page text is wrapped in an `UNTRUSTED PAGE DATA` block that the system prompt
  forbids obeying;
* injected payloads are asserted to remain inside that block (including a
  payload that tries to close the block early);
* page text is truncated before prompting;
* the model score is passed as an independent second opinion;
* a phishing verdict cannot be downgraded by injected instructions;
* if browsing fails, the URL score still drives the verdict (fail-safe).

---

## 8. Deployment (AWS)

```bash
# One-time: ECR repositories
aws ecr create-repository --repository-name phishing-api
aws ecr create-repository --repository-name phishing-sandbox

# Build & push (CI does this automatically on main via GitHub OIDC)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker build -f Dockerfile.api     -t "$REGISTRY/phishing-api:latest" .
docker build -f Dockerfile.sandbox -t "$REGISTRY/phishing-sandbox:latest" .
docker push "$REGISTRY/phishing-api:latest"
docker push "$REGISTRY/phishing-sandbox:latest"
```

Run on ECS Fargate as two services in the same task definition family:

| | API service | Sandbox service |
|---|---|---|
| Port | 8000 (behind an ALB) | 8080 (**service discovery / internal ALB only**) |
| Task role | S3 read for models, Bedrock invoke (if LLM enabled) | **none** |
| CPU / memory | 1 vCPU / 2 GB | 1 vCPU / 1 GB |
| Autoscaling | target tracking on ALB requests | target tracking on CPU |
| Secret | `SANDBOX_URL=http://sandbox.local:8080` | — |

Keep the sandbox in a private subnet with no inbound internet access, a
security group that only accepts traffic from the API service, `readonlyRootFilesystem`
plus a `tmpfs` for `/tmp`, and a 20-second task-level timeout. Model artifacts
are pulled from S3 into the API image or mounted at `/app/models`.

---

## 9. CI/CD

`.github/workflows/ci.yml` runs on every push/PR:

| Job | Contents |
|---|---|
| `test` | ruff lint + pytest with coverage gate |
| `data-validation` | schema/duplicate/invalid checks and the prepare stage |
| `safety` | sandbox-policy, certificate/domain and prompt-injection tests |
| `train` | full prepare → train → evaluate run, artifacts uploaded |
| `containers` | builds both Docker images with layer caching |
| `publish` | on `main`: builds and pushes both images to ECR via GitHub OIDC |

---

## 10. Monitoring

`GET /monitoring/stats` returns verdict distribution, latency p50/p95/max,
browser failure rate and mean model score over the last *N* analyses; every
analysis appends a JSON line to `data/monitoring/events.jsonl`.

`GET /monitoring/drift` computes PSI per URL feature between
`models/reference_features.csv` (training distribution) and recent live traffic,
flags features above `psi_threshold` and writes `reports/drift_report.json`. If
Evidently AI is installed, its `DataDriftPreset` summary is embedded in the same
report.

---

## 11. Tests

```bash
pytest -q                                  # everything
pytest -q tests/test_safety_and_injection.py  # injected-instruction resistance
pytest -q tests/test_checks.py                # sandbox policy + cert/domain checks
```

Coverage: URL features, page features and graceful degradation, data
validation, certificate/domain logic, sandbox safety policy, prompt injection,
agent orchestration (success, browser failure, no-model, URL-only), API contract
and monitoring/drift.

---

## 12. Expected outcome

A reproducible, containerized phishing-analysis system with a versioned dataset,
tracked experiments, a cloud-deployed API, automated tests and monitoring — in
which a sandboxed browser agent investigates the link itself and explains its
verdict. `reports/evaluation.json` quantifies how much the page evidence adds
over URL-only detection.

