# Environment
ENV ?= dev

# GPU / CPU
PLATFORM ?= gpu

# IMAGE REPOSITORY
IMAGE_REPO ?= atomsci/atomsci-ampl

# Jupyter Port
JUPYTER_PORT ?= 8888

# Python Executable
PYTHON_BIN ?= $(shell which python)

# Virtual Environment
VENV ?= venv

# Work Directory
WORK_DIR ?= work

.PHONY: build-docker install install-dev install-system install-venv jupyter-notebook jupyter-lab \
	pytest ruff ruff-fix setup uninstall uninstall-dev uninstall-system uninstall-venv

# Load Docker image
load-docker:
	docker load < ampl-$(PLATFORM)-$(ENV).tar.gz

# Pull Docker image
pull-docker:
	docker pull $(IMAGE_REPO):$(PLATFORM)-$(ENV)

# Push Docker image
push-docker:
	docker push $(IMAGE_REPO):$(PLATFORM)-$(ENV)

# Save Docker image
save-docker:
	docker save $(IMAGE_REPO):$(PLATFORM)-$(ENV) | gzip > ampl-$(PLATFORM)-$(ENV).tar.gz

# Build Docker image
build-docker:
	@echo "Building Docker image for $(PLATFORM)"
	docker buildx build -t $(IMAGE_REPO):$(PLATFORM)-$(ENV) --build-arg ENV=$(ENV) --platform linux/amd64 -f Dockerfile.$(PLATFORM) .

install: install-system

# Install atomsci-ampl in user space
install-dev:
	@echo "Installing atomsci-ampl for user"
	$(PYTHON_BIN) -m pip install -e . --user

# Install atomsci-ampl system-wide
install-system:
	@echo "Installing atomsci-ampl into $(PYTHON_BIN)"
	$(PYTHON_BIN) -m pip install -e .

# Install atomsci-ampl in virtual environment
install-venv:
	@echo "Installing atomsci-ampl into $(VENV)/"
	$(VENV)/bin/python -m pip install -e .

# Run Jupyter Notebook
jupyter-notebook:
	@echo "Starting Jupyter Notebook"
ifdef host
	docker run -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		--hostname $(host) \
		--privileged \
		-v $(shell pwd)/../$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(PLATFORM)-$(ENV) \
		/bin/bash -l -c "jupyter-notebook --ip=0.0.0.0 --no-browser --allow-root --port=$(JUPYTER_PORT)"
else
	docker run -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		-v $(shell pwd)/../$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(PLATFORM)-$(ENV) \
		/bin/bash -l -c "jupyter-notebook --ip=0.0.0.0 --no-browser --allow-root --port=$(JUPYTER_PORT)"
endif


# Run Jupyter Lab
jupyter-lab:
	@echo "Starting Jupyter Lab"
	docker run -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		-v $(shell pwd)/../$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(PLATFORM)-$(ENV) \
		/bin/bash -l -c "jupyter-lab --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"

# Run pytest
pytest:
	@echo "Running pytest"
	docker run -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		-v $(shell pwd)/$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(PLATFORM)-$(ENV) \
		/bin/bash -l -c "pytest"

# Run ruff linter
ruff:
	@echo "Running ruff"
	docker run -it $(IMAGE_REPO):$(PLATFORM)-$(ENV) /bin/bash -l -c "ruff check ."

# Run ruff linter with fix
ruff-fix:
	@echo "Running ruff with fix"
	docker run -it $(IMAGE_REPO):$(PLATFORM)-$(ENV) /bin/bash -l -c "ruff check . --fix"

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment with $(PLATFORM) dependencies"
	@echo "Removing old environment"
	rm -rf $(VENV)/ || true
	@echo "Creating new venv"
	python3.9 -m venv $(VENV)/
	$(VENV)/bin/pip install -U pip
	@echo "Installing dependencies"
	@if [ "$(PLATFORM)" = "gpu" ]; then \
		$(VENV)/bin/pip install -r pip/cuda_requirements.txt; \
	else \
		$(VENV)/bin/pip install -r pip/cpu_requirements.txt; \
	fi
	$(VENV)/bin/pip install -r pip/dev_requirements.txt
	$(MAKE) install-venv

uninstall: uninstall-system

# Uninstall atomsci-ampl from user space
uninstall-dev:
	@echo "Uninstalling atomsci-ampl for user"
	$(PYTHON_BIN) -m pip uninstall atomsci-ampl --user --yes

# Uninstall atomsci-ampl system-wide
uninstall-system:
	@echo "Uninstalling atomsci-ampl from $(PYTHON_BIN)"
	$(PYTHON_BIN) -m pip uninstall atomsci-ampl --yes

# Uninstall atomsci-ampl from virtual environment
uninstall-venv:
	@echo "Uninstalling atomsci-ampl from $(VENV)/"
	$(VENV)/bin/python -m pip uninstall atomsci-ampl --yes
