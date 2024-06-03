# Environment
ENV ?= dev

# Jupyter Port
JUPYTER_PORT ?= 8888

PLATFORM ?= gpu

# Python Executable
PYTHON_BIN ?= $(shell which python)

# IMAGE REPOSITORY
IMAGE_REPO ?= atomsci/atomsci-ampl

# Virtual Environment
VENV ?= venv

# Work Directory
WORK_DIR ?= work

.PHONY: build-docker install install-system install-venv jupyter-notebook jupyter-lab pytest ruff ruff-fix setup uninstall uninstall-system uninstall-venv

push-docker:
	docker push $(IMAGE_REPO):$(PLATFORM)-$(ENV)

build-docker:
	@echo "Building Docker image for $(PLATFORM)"
	docker build -t $(IMAGE_REPO):$(PLATFORM)-$(ENV) -f Dockerfile.$(PLATFORM) .

install: install-system

install-system:
	@echo "Installing atomsci-ampl into $(PYTHON_BIN)"
	$(PYTHON_BIN) -m pip install -e .

install-venv:
	@echo "Installing atomsci-ampl into $(VENV)/"
	$(VENV)/bin/python -m pip install -e .

# Run Jupyter Notebook
jupyter-notebook:
	@echo "Starting Jupyter Notebook"
	docker pull $(IMAGE_REPO)
	docker run -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		-v $(shell pwd)/$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO) \
		/bin/bash -l -c "jupyter-notebook --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"

# Run Jupyter Lab
jupyter-lab:
	@echo "Starting Jupyter Lab"
	docker pull $(IMAGE_REPO)
	docker run -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		-v $(shell pwd)/$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO) \
		/bin/bash -l -c "jupyter-lab --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"

# Run pytest
pytest:
	@echo "Running pytest"
	$(VENV)/bin/pytest atomsci/
	# docker run -it -v $(shell pwd)/$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO)-dev \
	# 	/bin/bash -l -c "pytest"

# Run ruff linter
ruff:
	@echo "Running ruff"
	$(VENV)/bin/ruff check .

# Run ruff linter with fix
ruff-fix:
	@echo "Running ruff with fix"
	$(VENV)/bin/ruff check . --fix

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment with $(PLATFORM) dependencies"
	rm -rf $(VENV)/ || true
	python3.9 -m venv $(VENV)/
	$(VENV)/bin/pip install -U pip
	@echo "Installing dependencies"
	@if [ "$(PLATFORM)" = "gpu" ]; then \
		$(VENV)/bin/pip install -r pip/cuda_requirements.txt; \
	else \
		$(VENV)/bin/pip install -r pip/cpu_requirements.txt; \
	fi
	$(VENV)/bin/pip install -r pip/dev_requirements.txt
	$(MAKE) install

uninstall: uninstall-system

uninstall-system:
	@echo "Uninstalling atomsci-ampl from $(PYTHON_BIN)"
	$(PYTHON_BIN) -m pip uninstall atomsci-ampl --yes

uninstall-venv:
	@echo "Uninstalling atomsci-ampl from $(VENV)/"
	$(VENV)/bin/python -m pip uninstall atomsci-ampl --yes
