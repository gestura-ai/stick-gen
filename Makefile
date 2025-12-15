
# Stick-Gen Makefile
# Unified command interface for local development and cloud (RunPod) operations.

PYTHON := PYTHONPATH=. python3
PIP := pip3
PYTEST := PYTHONPATH=. pytest

# Default target
.PHONY: help
help:
	@echo "Stick-Gen Command Interface"
	@echo "---------------------------"
	@echo "Setup:"
	@echo "  make setup            - Install dependencies and setup environment"
	@echo "  make clean            - Remove cache and temporary files"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-int         - Run integration tests only"
	@echo ""
	@echo "Local Development:"
	@echo "  make gen-data         - Generate synthetic training data (medium config)"
	@echo "  make train-local      - Run local training loop (medium config)"
	@echo "  make generate         - Generate a sample animation (.mp4)"
	@echo "  make generate-web     - Generate a sample with web export (.motion)"
	@echo ""
	@echo "Cloud (RunPod):"
	@echo "  make cloud-prep       - Prepare and upload data to S3"
	@echo "  make cloud-train-all  - Launch training for all models on RunPod"
	@echo "  make cloud-deploy     - Full deploy (setup vol, prep data, train)"
	@echo "  make cloud-check      - Check RunPod connection/credits"

# -------------------------------------------------------------------------
# Setup & Maintenance
# -------------------------------------------------------------------------

.PHONY: setup
setup:
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f output.mp4 output.motion

# -------------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------------

.PHONY: test
test:
	$(PYTEST) tests/

.PHONY: test-unit
test-unit:
	$(PYTEST) tests/unit/

.PHONY: test-int
test-int:
	$(PYTEST) tests/integration/

# -------------------------------------------------------------------------
# Local Workflows
# -------------------------------------------------------------------------

.PHONY: gen-data
gen-data:
	$(PYTHON) -m src.data_gen.dataset_generator --config configs/medium.yaml

.PHONY: train-local
train-local:
	$(PYTHON) -m src.train.train --config configs/medium.yaml

.PHONY: generate
generate:
	./stick-gen "A stick figure performing a karate kick" --output karate.mp4

.PHONY: generate-web
generate-web:
	# Generates output.mp4 AND output.motion
	./stick-gen "A breakdancer spinning on the floor" --output breakdance.mp4

# -------------------------------------------------------------------------
# Cloud Workflows (RunPod)
# -------------------------------------------------------------------------

# Ensure RunPod scripts are executable
.PHONY: _chmod-scripts
_chmod-scripts:
	chmod +x runpod/deploy.sh

.PHONY: cloud-prep
cloud-prep: _chmod-scripts
	./runpod/deploy.sh --prep-only

.PHONY: resume-upload
resume-upload: _chmod-scripts
	@if [ -z "$(VOLUME_ID)" ]; then echo "Error: VOLUME_ID is required. Run 'make resume-upload VOLUME_ID=...'"; exit 1; fi
	./runpod/resume_upload.sh --volume-id $(VOLUME_ID) --datacenter EU-CZ-1

.PHONY: cloud-train-all
cloud-train-all: _chmod-scripts
	./runpod/deploy.sh --datacenter EU-CZ-1 --models all

.PHONY: cloud-deploy
cloud-deploy: _chmod-scripts
	./runpod/deploy.sh --datacenter EU-CZ-1 --models all --full-deploy

.PHONY: cloud-check
cloud-check:
	@echo "Checking RunPod CLI..."
	runpodctl version
	@echo "Checking Balance..."
	runpodctl get user
