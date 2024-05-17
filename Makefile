# Work Directory
WORK_DIR ?= work

# Jupyter Port
JUPYTER_PORT ?= 8888

.PHONY: jupyter-notebook jupyter-lab

jupyter-notebook:
	docker pull atomsci/atomsci-ampl
	docker run -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -v $(pwd)/$(WORK_DIR):/$(WORK_DIR) atomsci/atomsci-ampl /bin/bash -l -c "jupyter-notebook --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"

jupyter-lab:
	docker pull atomsci/atomsci-ampl
	docker run -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -v $(pwd)/$(WORK_DIR):/$(WORK_DIR) atomsci/atomsci-ampl /bin/bash -l -c "jupyter-lab --ip=0.0.0.0 --allow-root --port=$(JUPYTER_PORT)"