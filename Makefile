# Work Directory
WORK_DIR ?= work

# Jupyter Port
JUPYTER_PORT ?= 8888

.PHONY: build-docker jupyter-notebook jupyter-lab

build-docker-cpu:
	docker build -t atomsci/atomsci-ampl/dev -f Dockerfile.cpu .

jupyter-notebook:
	docker pull atomsci/atomsci-ampl
	docker run -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -v $(pwd)/$(WORK_DIR):/$(WORK_DIR) atomsci/atomsci-ampl /bin/bash -l -c "jupyter-notebook --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"

jupyter-lab:
	docker pull atomsci/atomsci-ampl
	docker run -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -v $(pwd)/$(WORK_DIR):/$(WORK_DIR) atomsci/atomsci-ampl /bin/bash -l -c "jupyter-lab --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"

pytest:
	docker run -it -v $(pwd)/$(WORK_DIR):/$(WORK_DIR) atomsci/atomsci-ampl/dev /bin/bash -l "pytest"