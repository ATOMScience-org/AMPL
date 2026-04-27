# Architecture
ARCH ?=
# Conditionally set PLATFORM_ARG if ARCH is provided
ifneq ($(ARCH),)
    PLATFORM_ARG = --platform $(ARCH)
endif

# Environment
ENV ?= dev

# GPU / CPU, default is CPU
PLATFORM ?= cpu
ifeq ($(PLATFORM), gpu)
    GPU_ARG = --gpus all
endif

# define subtag. used for push
ifneq ($(ARCH),)
    SUBTAG = arm
else
    SUBTAG = $(PLATFORM)
endif

# Release version
VERSION=$(shell cat VERSION)

# If ENV is from master branch, we use VERSION for the tag, otherwise use PLATFORM
ifeq ($(ENV), prod)
    TAG = v$(VERSION)-$(SUBTAG)
else
    TAG = $(ENV)-$(PLATFORM)
endif

# IMAGE REPOSITORY
IMAGE_REPO ?= atomsci/atomsci-ampl

# Jupyter Port
JUPYTER_PORT ?= 8888

# Python Executable
PYTHON_BIN ?= $(shell which python)

# Virtual Environment
VENV ?= atomsci-env

# Work Directory
WORK_DIR ?= work

.PHONY: update-lock-cpu update-lock-cuda update-lock-rocm update-lock-mchip sync-cpu sync-cuda sync-rocm sync-mchip  \
        build-docker install install-dev install-system install-venv jupyter-notebook jupyter-lab \
	pytest ruff ruff-fix setup

# Load Docker image
load-docker:
	docker load < ampl-$(TAG).tar.gz

# Pull Docker image
pull-docker:
	docker pull $(IMAGE_REPO):$(TAG)

# Push Docker image
push-docker:
	docker buildx build --no-cache -t $(IMAGE_REPO):$(TAG) -t $(IMAGE_REPO):latest-$(SUBTAG) --build-arg ENV=$(ENV) $(PLATFORM_ARG) --push -f Dockerfile.$(PLATFORM) .

# Save Docker image
save-docker:
	docker save $(IMAGE_REPO):$(TAG) | gzip > ampl-$(TAG).tar.gz

# Build Docker image
build-docker:
	@echo "Building Docker image for $(PLATFORM) <cpu|gpu|rocm|arm>"
	docker buildx build -t $(IMAGE_REPO):$(TAG) --build-arg ENV=$(ENV) $(PLATFORM_ARG) --load -f Dockerfile.$(PLATFORM) .

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
	  $(GPU_ARG) \
		--hostname $(host) \
		--privileged \
		-v $(shell pwd)/../$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(TAG) \
		/bin/bash -l -c "jupyter-notebook --ip=0.0.0.0 --no-browser --allow-root --port=$(JUPYTER_PORT)"
else
	docker run -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		$(GPU_ARG) \
		-v $(shell pwd)/../$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(TAG) \
		/bin/bash -l -c "jupyter-notebook --ip=0.0.0.0 --no-browser --allow-root --port=$(JUPYTER_PORT)"
endif


# Run Jupyter Lab
jupyter-lab:
	@echo "Starting Jupyter Lab"
	docker run -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
		-v $(shell pwd)/../$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(TAG) \
		/bin/bash -l -c "jupyter-lab --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"

# Run pytest
pytest: pytest-unit pytest-integrative

pytest-integrative:
	@echo "Running integrative tests"
	docker run -v $(shell pwd)/$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(TAG) \
			/bin/bash -l -c "cd atomsci/ddm/test/integrative && ./integrative_batch_tests.sh"

pytest-unit:
	@echo "Running unit tests"
	docker run -v $(shell pwd)/$(WORK_DIR):/$(WORK_DIR) $(IMAGE_REPO):$(TAG) \
		       /bin/bash -l -c "cd atomsci/ddm/test/unit && python3.10 -m pytest --capture=sys --capture=fd --cov=atomsci -vv"

# Run ruff linter
ruff:
	@echo "Running ruff"
	docker run -it $(IMAGE_REPO):$(TAG) /bin/bash -l -c "ruff check ."

# Run ruff linter with fix
ruff-fix:
	@echo "Running ruff with fix"
	docker run -it $(IMAGE_REPO):$(TAG) /bin/bash -l -c "ruff check . --fix"

shell:
	docker run -v $(shell pwd)/../$(WORK_DIR):/$(WORK_DIR) -it $(IMAGE_REPO):$(TAG) /bin/bash

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment with $(PLATFORM) dependencies"
	@echo "Removing old environment"
	rm -rf $(VENV)/ || true
	@echo "Creating new venv"
	python3.10 -m venv $(VENV)/
	$(VENV)/bin/pip install -U pip
	@echo "Installing dependencies"
	@if [ "$(PLATFORM)" = "gpu" ]; then \
		$(VENV)/bin/pip install -r pip/cuda_requirements.txt; \
	else \
		$(VENV)/bin/pip install -r pip/cpu_requirements.txt; \
	fi
	$(VENV)/bin/pip install -r pip/dev_requirements.txt
	$(MAKE) install-venv

# for uv env

# use these to create/update a lock env. for maintainers only
update-lock-cpu:
	./update_uv_lock.sh cpu

update-lock-cuda:
	./update_uv_lock.sh cuda

update-lock-rocm:
	./update_uv_lock.sh rocm

update-lock-mchip:
	./update_uv_lock.sh mchip

# use the matching lockfile to create or refresh .venv-<platform>. for general users
sync-cpu:
	./sync_uv_env.sh cpu

sync-cuda:
	./sync_uv_env.sh cuda

sync-rocm:
	./sync_uv_env.sh rocm

sync-mchip:
	./sync_uv_env.sh mchip