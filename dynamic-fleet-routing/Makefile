.PHONY: install test train evaluate tune benchmark docker run clean lint

# Installation
install:
	pip install -e ".[all]"

install-base:
	pip install -e .

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# Training
train:
	python -m src.training.train_ppo --config configs/ppo.yaml

train-small:
	python -m src.training.train_ppo --config configs/ppo.yaml --total-timesteps 50000

# Evaluation
evaluate:
	python -m src.training.evaluate --config configs/base.yaml

# Hyperparameter Tuning
tune:
	python -m src.training.tune --config configs/tuning.yaml

# Benchmarking
benchmark:
	python scripts/benchmark_latency.py

# Experiments
experiment:
	python scripts/run_experiment.py

# Dataset Generation
dataset:
	python scripts/generate_dataset.py

# API Server
run:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker:
	docker compose up --build

docker-build:
	docker build -t dynamic-fleet-routing .

# MLflow UI
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
	rm -rf mlruns/ 2>/dev/null || true

help:
	@echo "Available commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make train        - Train PPO agent"
	@echo "  make train-small  - Quick training smoke test"
	@echo "  make evaluate     - Evaluate all methods"
	@echo "  make tune         - Run hyperparameter tuning"
	@echo "  make benchmark    - Run latency benchmarks"
	@echo "  make experiment   - Run ablation experiments"
	@echo "  make run          - Start FastAPI server"
	@echo "  make docker       - Build and run Docker"
	@echo "  make mlflow-ui    - Start MLflow UI"
	@echo "  make clean        - Remove build artifacts"
