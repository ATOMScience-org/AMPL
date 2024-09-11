## Makefile for Managing Jupyter Environments and Docker Images

### **Overview**

This Makefile is designed to manage Jupyter environments, Docker images, and the installation of dependencies for the `atomsci-ampl` project. It simplifies the process of building, pulling, pushing Docker images, and running Jupyter servers.

### **Configuration Variables**

- **Environment Configuration:**

  - `ENV`: Specifies the environment (`dev`, `prod`, etc.). Default is `dev`.
  - `PLATFORM`: Specifies the platform (`gpu`, `cpu`, `arm`). Default is `cpu`. `arm` for macOS arm.
  - `ARCH`: Specifies the platform architecture (`linux/amd64` or `linux/arm64`) to use.  Default to current platform.
  - `VERSION`: The version to be used for the docker tag.

- **Docker Configuration:**

  - `IMAGE_REPO`: Specifies the Docker image repository. Default is `atomsci/atomsci-ampl`.

- **Jupyter Configuration:**

  - `JUPYTER_PORT`: Specifies the port on which Jupyter will run. Default is `8888`.

- **Python Configuration:**

  - `PYTHON_BIN`: Specifies the Python executable to use.
  - `VENV`: Specifies the virtual environment directory. Default is `atomsci-env`.

- **Work Directory:**
  - `WORK_DIR`: Specifies the working directory inside the Docker container where files will be stored. Default is `work`.

### **Makefile Targets**

- **Docker Image Management:**

  - `make pull-docker`: Pull the Docker image from the repository.
  - `make push-docker`: Push the Docker image to the repository.
  - `make load-docker`: Load a Docker image from a tarball (`ampl-$(PLATFORM)-$(ENV).tar.gz`).
  - `make save-docker`: Save the Docker image to a tarball (`ampl-$(PLATFORM)-$(ENV).tar.gz`).
  - `make build-docker`: Build the Docker image for the specified platform and environment.

- **Installation:**

  - `make install`: Install `atomsci-ampl` system-wide.
  - `make install-dev`: Install `atomsci-ampl` in user space.
  - `make install-system`: Install `atomsci-ampl` system-wide.
  - `make install-venv`: Install `atomsci-ampl` in a virtual environment.

- **Jupyter Notebook and Lab:**

  - `make jupyter-notebook`: Start a Jupyter Notebook server.
  - `make jupyter-lab`: Start a Jupyter Lab server.

- **Testing and Linting:**

  - `make pytest`: Run all tests within the Docker container.
  - `make pytest-unit`: Run unit tests within the Docker container.
  - `make pytest-integrative`: Run integrative tests within the Docker container.
  - `make ruff`: Run the `ruff` linter.
  - `make ruff-fix`: Run the `ruff` linter and automatically fix issues.

- **Entrypoint**

  - `make shell`: Go inside the container's shell.

- **Setup and Uninstallation:**
  - `make setup`: Set up a virtual environment and install dependencies.
  - `make uninstall`: Uninstall `atomsci-ampl` system-wide.
  - `make uninstall-dev`: Uninstall `atomsci-ampl` from user space.
  - `make uninstall-system`: Uninstall `atomsci-ampl` system-wide.
  - `make uninstall-venv`: Uninstall `atomsci-ampl` from the virtual environment.

### **Usage Examples**

#### **Managing Docker Images**

To load a Docker image from a tarball:

```bash
make load-docker
```

To pull the latest Docker image:

```bash
make pull-docker
```

To build a Docker image:

```bash
make build-docker
```

#### **Starting Jupyter Servers**

To start a Jupyter Notebook server:

```bash
make jupyter-notebook
```

To start a Jupyter Lab server:

```bash
make jupyter-lab
```

#### **Setting Up the Development Environment**

To set up a virtual environment with the appropriate dependencies:

```bash
make setup
```

This Makefile provides a comprehensive approach to managing Docker images, running Jupyter servers, and handling the installation and setup of your development environment. It streamlines workflows, making it easier to maintain and develop the `atomsci-ampl` project.
