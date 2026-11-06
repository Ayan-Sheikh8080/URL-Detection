.PHONY: help setup install train test docker-build docker-push e2e-test lint

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "} {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	@npm install
	@python3 -m pip install -r requirements-dev.txt

install: ## Alias for setup

train: ## Train ML models
	@python3 -m src.training.train

test: ## Run unit tests
	@npm test

e2e-test: ## Run end-to-end tests
	@npm run test:e2e

lint: ## Run linter
	@npm run lint

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