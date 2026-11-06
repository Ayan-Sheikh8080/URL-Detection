.PHONY: help setup install prepare train test docker-build docker-push e2e-test lint

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "} {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	@npm install
	@PYYAML_FORCE_LIBYAML=0 python -m pip install -r requirements.txt
	@python -m playwright install chromium

install: setup ## Alias for setup

prepare: ## Validate and prepare the training data
	@python -m src.data.prepare --strict

train: ## Train ML models
	@python -m src.training.train

test: ## Run unit tests
	@npm test
	@python -m pytest -q

e2e-test: ## Run end-to-end tests
	@python -m pytest -q tests/python/test_api.py

lint: ## Run linter
	@npm run lint
	@python -m compileall -q src tests

docker-build: ## Build Docker images
	@docker build -f Dockerfile.api -t phishing-api:latest .
	@docker build -f Dockerfile.sandbox -t phishing-sandbox:latest .

docker-push: ## Push Docker images to ECR
	@aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin
	@docker tag phishing-api:latest $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_API_REPO):latest
	@docker tag phishing-sandbox:latest $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_SANDBOX_REPO):latest
	@docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_API_REPO):latest
	@docker push $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_SANDBOX_REPO):latest

docker-compose-up: ## Start full stack with docker compose
	@docker compose up --build

docker-compose-down: ## Stop full stack
	@docker compose down