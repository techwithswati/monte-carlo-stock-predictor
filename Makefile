# =============================================================================
# Makefile — Monte Carlo Stock Predictor
# Usage: make help
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test test-cov \
        run-api run-sim docker-build docker-up docker-down \
        clean publish

PYTHON     := python3
PIP        := pip3
IMAGE_NAME := monte-carlo-stock-predictor
PORT       := 8000

# ── Colours ──────────────────────────────────────────────────────────────────
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

help: ## Show this help
	@echo "$(BOLD)Monte Carlo Stock Predictor$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install all dependencies including dev tools
	$(PIP) install -r requirements.txt -r requirements-dev.txt

# ── Code quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check src/ tests/ run_simulation.py

format: ## Auto-format with black + ruff --fix
	black src/ tests/ run_simulation.py
	ruff check --fix src/ tests/ run_simulation.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test: ## Run unit tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

test-fast: ## Run tests excluding slow integration tests
	pytest tests/ -v -m "not slow"

# ── Run ───────────────────────────────────────────────────────────────────────
run-api: ## Start the FastAPI server locally (dev mode)
	PYTHONPATH=. uvicorn src.api.app:app --reload --host 0.0.0.0 --port $(PORT)

run-sim: ## Run CLI simulation (AAPL, GBM, 10k sims)
	PYTHONPATH=. $(PYTHON) run_simulation.py \
		--ticker AAPL \
		--model gbm \
		--sims 10000 \
		--days 252

run-sim-heston: ## Run Heston stochastic vol simulation
	PYTHONPATH=. $(PYTHON) run_simulation.py \
		--ticker NVDA \
		--model heston \
		--sims 5000 \
		--days 126

run-sim-jump: ## Run Merton Jump-Diffusion simulation
	PYTHONPATH=. $(PYTHON) run_simulation.py \
		--ticker TSLA \
		--model jump_diffusion \
		--sims 5000

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME):latest .

docker-up: ## Start full stack (API + Prometheus + Grafana)
	docker compose -f infrastructure/docker-compose.yml up -d
	@echo "$(GREEN)API:        http://localhost:8000/docs$(RESET)"
	@echo "$(GREEN)Prometheus: http://localhost:9090$(RESET)"
	@echo "$(GREEN)Grafana:    http://localhost:3000 (admin/montecarlo)$(RESET)"

docker-down: ## Stop all containers
	docker compose -f infrastructure/docker-compose.yml down

docker-logs: ## Tail API container logs
	docker compose -f infrastructure/docker-compose.yml logs -f api

docker-shell: ## Open shell in running API container
	docker compose -f infrastructure/docker-compose.yml exec api /bin/bash

# ── Kubernetes ────────────────────────────────────────────────────────────────
k8s-apply: ## Apply Kubernetes manifests
	kubectl apply -f infrastructure/k8s/

k8s-status: ## Check deployment status
	kubectl get pods,svc,hpa -n finance

k8s-logs: ## Tail production pod logs
	kubectl logs -l app=mc-stock-predictor -n finance --follow

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts, caches, outputs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf outputs/ .coverage coverage.xml
	@echo "$(GREEN)Clean!$(RESET)"

# ── Publish ───────────────────────────────────────────────────────────────────
publish: ## Push image to GitHub Container Registry
	@echo "$(BOLD)Building multi-arch image...$(RESET)"
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		-t ghcr.io/$(GITHUB_USERNAME)/$(IMAGE_NAME):latest \
		--push .
