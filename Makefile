.PHONY: help install install-dev data train evaluate test safety lint api sandbox up down clean dvc-repro

help:
	@echo "Targets:"
	@echo "  install      install runtime dependencies"
	@echo "  install-dev  install dev dependencies and git hooks"
	@echo "  data         run the data preparation stage"
	@echo "  train        train and compare all model configurations (MLflow)"
	@echo "  evaluate     evaluate promoted models"
	@echo "  dvc-repro    reproduce the full DVC pipeline"
	@echo "  test         run the unit test suite"
	@echo "  safety       run safety + prompt-injection tests"
	@echo "  lint         run ruff"
	@echo "  api          serve the FastAPI app locally"
	@echo "  sandbox      serve the browser sandbox locally"
	@echo "  up/down      start/stop the docker-compose stack"
	@echo "  clean        remove generated artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

data:
	python -m src.data.prepare

train:
	python -m src.training.train

evaluate:
	python -m src.training.evaluate

dvc-repro:
	dvc repro

test:
	pytest -q

safety:
	pytest -q tests/test_checks.py tests/test_safety_and_injection.py -v

lint:
	ruff check src tests --line-length 110

api:
	uvicorn src.serving.api:app --reload --port 8000

sandbox:
	uvicorn src.sandbox.server:app --reload --port 8080

up:
	docker compose up --build -d

down:
	docker compose down

clean:
	rm -rf mlruns mlflow.db .pytest_cache .ruff_cache reports/*.json
	rm -rf data/processed/* data/monitoring/* models/*.pkl models/*.joblib models/index.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
