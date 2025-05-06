#!/usr/bin/make -f

.PHONY: setup run dev test docker docker-build docker-run clean help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
APP_DIR := app
PORT := 5005
HOST := 0.0.0.0
ENV_FILE := .env

# Default target
help:
	@echo "Genesilico OCR + AI Agent Service"
	@echo ""
	@echo "Usage:"
	@echo "  make setup           Install dependencies and create directories"
	@echo "  make run             Run the application in production mode"
	@echo "  make dev             Run the application in development mode with auto-reload"
	@echo "  make test            Run tests"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-run      Run Docker container"
	@echo "  make clean           Clean build artifacts and caches"
	@echo "  make help            Show this help message"

# Setup
setup:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Creating data directories..."
	mkdir -p data/documents data/ocr_results data/trf_outputs
	@echo "Creating environment file..."
	$(PYTHON) scripts/create_env.py --force
	@echo "Setup complete. Remember to update $(ENV_FILE) with your API keys."

# Run
run:
	@echo "Running the application in production mode..."
	$(PYTHON) run.py --env production

# Dev
dev:
	@echo "Running the application in development mode..."
	$(PYTHON) run.py --env development --reload

# Test
test:
	@echo "Running tests..."
	pytest tests/

# Docker build
docker-build:
	@echo "Building Docker image..."
	docker build -t genesilico-ocr -f docker/Dockerfile .

# Docker run
docker-run:
	@echo "Running Docker container..."
	docker run --name genesilico-ocr -p $(PORT):$(PORT) -d genesilico-ocr

# Clean
clean:
	@echo "Cleaning build artifacts and caches..."
	rm -rf __pycache__
	rm -rf $(APP_DIR)/__pycache__
	rm -rf $(APP_DIR)/*/__pycache__
	rm -rf $(APP_DIR)/*/*/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	@echo "Clean complete."
