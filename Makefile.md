## Makefile for Jupyter Environments

### **Overview**

This Makefile is designed to simplify the management of Jupyter environments within Docker containers. It includes commands to run both Jupyter Notebook and Jupyter Lab.

### **Configuration Variables**

- `WORK_DIR`: Specifies the working directory inside the Docker container where files will be stored. Default is `work`.
- `JUPYTER_PORT`: Specifies the port on which Jupyter will run. Default is `8888`.

### **Jupyter Notebook**

- `make jupyter-notebook`: Starts a Jupyter Notebook server.
  - Pulls the latest `atomsci/atomsci-ampl` Docker image.
  - Runs a Docker container with the specified port and working directory.
  - Starts the Jupyter Notebook server accessible via the specified port.

### **Jupyter Lab**

- `make jupyter-lab`: Starts a Jupyter Lab server.
  - Pulls the latest `atomsci/atomsci-ampl` Docker image.
  - Runs a Docker container with the specified port and working directory.
  - Starts the Jupyter Lab server accessible via the specified port.

### **Usage Example**

To start a Jupyter Notebook server:

```bash
make jupyter-notebook
```
