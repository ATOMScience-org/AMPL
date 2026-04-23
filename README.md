[![Develop test](https://github.com/ATOMScience-org/AMPL/actions/workflows/pytest.yml/badge.svg)](https://github.com/ATOMScience-org/AMPL/actions/workflows/pytest.yml)  [![Linter](https://github.com/ATOMScience-org/AMPL/actions/workflows/lint.yml/badge.svg)](https://github.com/ATOMScience-org/AMPL/actions/workflows/lint.yml)    [![Documentation Status](https://readthedocs.org/projects/ampl/badge/?version=stable)](https://ampl.readthedocs.io/en/latest/?badge=stable)
 [![codecov](https://codecov.io/gh/ATOMScience-org/AMPL/graph/badge.svg)](https://codecov.io/gh/ATOMScience-org/AMPL)

![GitHub Release](https://img.shields.io/github/v/release/ATOMScience-org/AMPL)  [![License](http://img.shields.io/:license-mit-blue.svg)](https://github.com/ATOMScience-org/AMPL/blob/master/LICENSE)   [![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/atomscience) [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?logo=YouTube&logoColor=white)](https://www.youtube.com/channel/UCOF6zZ7ltGwopYCoOGIFM-w)

| [Install](#install) | [Docker](#install-with-docker) | [Tutorials](#ampl-tutorials) |  [Features](#ampl-features) | [Pipeline parameters](atomsci/ddm/docs/PARAMETERS.md) | [Docs](https://ampl.readthedocs.io/en/latest/) |

# ATOM Modeling PipeLine (AMPL) for Drug Discovery
An open-source, end-to-end software pipeline for data curation, model building, and molecular property prediction to advance in silico drug discovery.

*Created by the [Accelerating Therapeutics for Opportunities in Medicine (ATOM) Consortium](https://atomscience.org)*

<p align="center">
  <img src="assets/ATOM_cymatics_black_wordmark.jpg" width="370" height="100" alt="AMPL logo">
</p>

The ATOM Modeling PipeLine (AMPL) extends the functionality of DeepChem and supports an array of machine learning and molecular featurization tools to predict key potency, safety and pharmacokinetic-relevant parameters. AMPL has been benchmarked on a large collection of pharmaceutical datasets covering a wide range of parameters. This is a living software project with active development. Check back for continued updates. Feedback is welcomed and appreciated, and the project is open to contributions! An [article describing the AMPL project](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b01053) was published in JCIM. The AMPL pipeline documentation is available [here](https://ampl.readthedocs.io/en/latest/pipeline.html).

Check out our new tutorial series that walks through AMPL's end-to-end modeling pipeline to build a machine learning model! View them in our [docs](https://ampl.readthedocs.io/en/latest/) or as Jupyter notebooks in our [repo](https://github.com/ATOMScience-org/AMPL/tree/master/atomsci/ddm/examples/tutorials).

![Static Badge](https://img.shields.io/badge/Announcement-1.7.0-blue)

In addition to our written tutorials, we now provide a series of video tutorials on our YouTube channel, [ATOMScience-org](https://www.youtube.com/channel/UCOF6zZ7ltGwopYCoOGIFM-w). These videos are created to assist users in exploring and leveraging AMPL's robust capabilities. We provided a playlist for easy streamlined Learning:

[![AMPL Tutorial Playlist](https://img.shields.io/badge/AMPL_Tutorial_Playlist-%23FF0000.svg?logo=YouTube&logoColor=white)](https://www.youtube.com/playlist?list=PLe85Q-Gf8eFgYGQmUDSTlSjJorQZyDG8E)

---
## Table of contents
- [Installation](#installation)
  - [Set up uv](#setup-uv)
    - [What is `uv`?](#what-is-uv)
    - [Repository helper scripts](#repository-helper-scripts)
    - [Requirements](#requirements)
    - [Install `uv`](#install-uv)
    - [Create and activate an environment](#create-and-activate-an-environment)
    - [Platform-specific setup](#platform-specific-setup)
    - [Troubleshooting](#troubleshooting)
      - [`uv` not found](#uv-not-found)
      - [Missing lockfile](#missing-lockfile)
      - [Wrong Python/Pytest](#wrong-pythonpytest)
      - [Library not found after activation](#library-not-found-after-activation)
      - [Package import fails or environment out of sync](#package-import-fails-or-environment-out-of-sync)
  - [Install AMPL](#install-ampl)
    - [Install from PyPI](#install-from-pypi)
    - [Install from a local clone for development](#install-from-a-local-clone-for-development)
    - [Build and install from a local clone](#build-and-install-from-a-local-clone)
- [AMPL Features](#ampl-features)
- [Running AMPL](#running-ampl)
- [Tests](#tests)
- [Advanced AMPL usage](#advanced-ampl-usage)
- [Advanced testing](#advanced-testing)
- [Tutorials](#ampl-tutorials)
- [Development](#development)
- [Project information](#project-information)
- [Suggestions or Report Issues](#support-suggestions-or-report-issues)

## Useful links
- [Pipeline parameters (options)](atomsci/ddm/docs/PARAMETERS.md)
- [Library documentation](https://ampl.readthedocs.io/en/latest/index.html)

---

## Installation

AMPL 1.8 supports `Python 3.10` on CPU systems and CUDA-enabled Linux systems using CUDA 11.8. All other systems are experimental. For a quick install summary, see [here](#install-summary). For more information, see [DeepChem](https://deepchem.readthedocs.io/en/latest/get_started/installation.html), [TensorFlow](https://www.tensorflow.org/install/pip), [PyTorch](https://pytorch.org/get-started/locally/), and [DGL](https://www.dgl.ai/pages/start.html).

For installation on Apple Silicon M chips, see the Docker container instructions.

AMPL uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management.

### Set up `uv`

#### What is `uv`?

`uv` is a fast Python tool used in this project to:

| Use | Command |
|---|---|
| Create virtual environments | `uv venv` |
| Sync dependencies from `pyproject.toml` | `uv sync` |
| Install packages into the environment | `uv pip install` |

#### Repository helper scripts

This repository includes helper commands for managing platform-specific `uv` environments and lockfiles.

| Command | Purpose | When to use it |
|---|---|---|
| `./update_uv_lock.sh <platform>` | Regenerate a platform-specific lockfile | Run when dependencies change, for example after editing `pyproject.toml` |
| `./sync_uv_env.sh <platform>` | Create or rebuild a local platform environment from an existing platform lockfile | Run the first time you set up `.venv-<platform>`, or anytime you want a clean rebuild |
| `make sync-<platform>` | Sync an existing local environment | Run when `.venv-<platform>` already exists and you want to refresh it for local work |
| `source .venv-<platform>/bin/activate` | Activate the virtual environment in your shell | Run after the environment has been created or synced |

These commands map to the platform-specific environments and lockfiles below:

| Platform | Virtual environment | Lockfile |
|---|---|---|
| CPU | `.venv-cpu` | `uv.lock.cpu` |
| CUDA | `.venv-cuda` | `uv.lock.cuda` |
| ROCm | `.venv-rocm` | `uv.lock.rocm` |
| Apple Silicon / M chip | `.venv-mchip` | `uv.lock.mchip` |

In practice:

- Run `./update_uv_lock.sh <platform>` only when dependency definitions change and the platform lockfile must be regenerated.
- Run `./sync_uv_env.sh <platform>` when creating a platform environment for the first time, or when rebuilding it from scratch.
- Run `make sync-<platform>` only for routine syncing of an existing local environment.
- Run `source .venv-<platform>/bin/activate` to use the environment in your shell.

> **Note:** Most users should start with `./sync_uv_env.sh <platform>`. `update_uv_lock.sh` is mainly for maintainers updating lockfiles.

#### Requirements

| Item | Requirement |
|---|---|
| Python | 3.10 |
| Supported range | `>=3.10,<3.11` |
| Platforms | `cpu`, `cuda`, `rocm`, `mchip` |

#### Install `uv`

Install `uv` using one of the following methods:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or
```bash
pip install uv
```
Verify installation:
```bash
uv --version
```

#### Create and activate an environment

Use `sync_uv_env.sh` to create or rebuild the environment for your target platform from the committed platform lockfile.

| Platform | Environment | Create or rebuild | Activate command |
|---|---|---|---|
| CPU | `.venv-cpu` | `./sync_uv_env.sh cpu` | `source .venv-cpu/bin/activate` |
| CUDA | `.venv-cuda` | `./sync_uv_env.sh cuda` | `source .venv-cuda/bin/activate` |
| ROCm | `.venv-rocm` | `./sync_uv_env.sh rocm` | `source .venv-rocm/bin/activate` |
| Apple Silicon / M chip | `.venv-mchip` | `./sync_uv_env.sh mchip` | `source .venv-mchip/bin/activate` |

If you already have the environment and only want to refresh it, you may use:

| Platform | Environment | Refresh command |
|---|---|---|
| CPU | `.venv-cpu` | `make sync-cpu` |
| CUDA | `.venv-cuda` | `make sync-cuda` |
| ROCm | `.venv-rocm` | `make sync-rocm` |
| Apple Silicon / M chip | `.venv-mchip` | `make sync-mchip` |

Example, first-time setup:

```bash
./sync_uv_env.sh cpu
source .venv-cpu/bin/activate
```

If `.venv-cpu` already exists and you only want to refresh it:
```bash
make sync-cpu
source .venv-cpu/bin/activate
```
#### Platform-specific setup

The following settings may be useful depending on your platform and runtime environment.

| Platform | Optional setup | Useful environment variables |
|---|---|---|
| CPU | none | `export OPENBLAS_NUM_THREADS=1` |
| CUDA | load site CUDA module if required | `module load cuda`<br>`export CUDA_HOME=/usr/local/cuda`<br>`export PATH="$CUDA_HOME/bin:$PATH"`<br>`export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"` |
| ROCm | site-specific ROCm setup if required | `module load rocm`<br> `export ROCM_HOME="$(dirname "$(dirname "$(readlink -f "$(which hipcc)")")")"`<br>`export PATH="$ROCM_HOME/bin:$PATH"` |
| Apple Silicon / M chip | none | usually none required |

#### Troubleshooting

##### `uv` not found

Install `uv`, then verify:
```bash
uv --version
```
If needed, start a new shell session or update your `PATH`.

##### Missing lockfile

After `uv sync`, if you see:

```bash
Missing lockfile: uv.lock.<platform>
```
the platform lockfile is not present in your checkout. Confirm that you are on the correct branch and that the lockfile has been committed.

If the lockfile is genuinely missing, contact the maintainers or regenerate it only if you are updating dependencies.

##### Wrong Python/Pytest

Check that Python 3.10 is being used:
```bash
which python
python --version
which pytest
```
They should come from the virtual environment, for example:
```bash
.venv-cpu/bin/python
.venv-cpu/bin/pytest
```
##### Library not found after activation

Confirm the correct environment is active:
```bash
which python
echo $VIRTUAL_ENV
```
If needed, set:
```bash
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
```
##### Package import fails or environment out of sync

Try:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import rdkit; print('rdkit ok')"
```
If imports fail, resync the environment:
```bash
make sync-<platform>
source .venv-<platform>/bin/activate
```
---
### Install AMPL

Since `1.8.0`, AMPL is published to PyPI. Users can install it directly, without building it locally.

> **Note:** Please ensure `Python 3.10` is used or loaded for the package to run. For LLNL users on LC, run `module load python/3.10.8`.

#### Install from PyPI

Install the published package directly from PyPI:

```
uv pip install atomsci-ampl
```

#### Install from a local clone for development

If you want to develop AMPL locally, clone the repository and install it in editable mode:

```
git clone https://github.com/ATOMScience-org/AMPL.git
cd AMPL
./install-dev.sh
```

This installs the package from the local source tree, so changes to the code are available without reinstalling.

#### Build and install from a local clone

If you want to build the package locally and then install the built artifact:

```
git clone https://github.com/ATOMScience-org/AMPL.git
cd AMPL
./build.sh
./install.sh
```

#### *(Optional) LLNL LC only*: if you use [model_tracker](https://ampl.readthedocs.io/en/latest/pipeline.html#module-pipeline.model_tracker), install `atomsci.clients`
```bash
# LLNL only: required for ATOM model_tracker
pip install -r clients_requirements.txt
```
---

## Create jupyter notebook kernel (optional)

To run AMPL from Jupyter Notebook, first activate your environment and then run:
```bash
python -m ipykernel install --user --name atomsci-env
```
---
#### *(Optional) LLNL LC only*: if you use [model_tracker](https://ampl.readthedocs.io/en/latest/pipeline.html#module-pipeline.model_tracker), install atomsci.clients
```bash
# LLNL only: required for ATOM model_tracker
pip install -r clients_requirements.txt
```

---
## Create jupyter notebook kernel (optional)
To run AMPL from Jupyter Notebook. To setup a new kernel, first activate your environment and then run the following command:

```
python -m ipykernel install --user --name atomsci-env
```

---
## Install with Docker
- Download and install Docker Desktop.
  - https://www.docker.com/get-started
- Create a workspace folder to mount with Docker environment and transfer files.
- Get the Docker image and run it. Since 1.6.3, there are some changes with the AMPL Docker.

To retrieve, run version 1.6.2 or earlier, please specify the desired version tag:
  ```
  docker pull atomsci/atomsci-ampl:v1.6.2
  docker run -it -p 8888:8888 -v </local_workspace_folder>:</directory_in_docker> atomsci/atomsci-ampl:v1.6.2
  ```

For AMPL versions 1.6.3 and later, we offer downloadable images for various platforms (CPU, GPU or Linux/ARM64). To run a Docker container, be sure to append `bash` at the end of the command to open a bash session.

  ```
  docker pull atomsci/atomsci-ampl:latest-<platform> # can be cpu, gpu, or arm (for arm64 chip)
  docker run -it -p 8888:8888 -v </local_workspace_folder>:</directory_in_docker> atomsci/atomsci-ampl:latest-<platform> bash
  #inside docker environment
  jupyter-notebook --ip=0.0.0.0 --allow-root --port=8888 &
  # -OR-
  jupyter-lab --ip=0.0.0.0 --allow-root --port=8888 &
  ```
- Visit the provided URL in your browser, ie
  - http://d33b0faf6bc9:8888/?token=656b8597498b18db2213b1ec9a00e9d738dfe112bbe7566d
  - Replace the "d33b0faf6bc9" with "localhost"
  - If this doesn't work, exit the container and change port from 8888 to some other number such as 7777 or 8899 (in all 3 places it's written), then rerun both commands
- From the notebook, you may need to set the kernel that atomsci is installed ("atomsci-venv") in order to acccess the `atomsci` package.

For additional options related to building, running, and other Docker development tasks, please refer to [Makefile.md](Makefile.md).

---

To remove an entire virtual environment named "atomsci-env":
```bash
rm -rf $ENVROOT/atomsci-env
```

---
## AMPL Features
<details><summary>AMPL enables tasks for modeling and prediction from data ingestion to data analysis and can be broken down into the following stages:</summary>

### 1. Data curation
- Generation of RDKit molecular SMILES structures
- Processing of qualified or censored data processing
- Curation of activity and property values

### 2. Featurization
- Extended connectivity fingerprints (ECFP)
- Graph convolution latent vectors from DeepChem
- Chemical descriptors from Mordred package
- Descriptors generated by MOE (requires MOE license)

### 3. Model training and tuning
- Test set selection
- Cross-validation
- Uncertainty quantification

### 4. Supported models
- scikit-learn random forest models
- XGBoost models
- Fully connected neural networks
- Graph convolution models

### 5. Visualization and analysis
- Visualization and analysis tools
</details>
Details of running specific features are within the [parameter (options) documentation](#pipeline-parameters). More detailed documentation is in the [library documentation](#library-documentation).

---
## Running AMPL
AMPL can be run from the command line or by importing into Python scripts and Jupyter notebooks.

### Python scripts and Jupyter notebooks
AMPL can be used to fit and predict molecular activities and properties by importing the appropriate modules. See the [examples](atomsci/ddm/examples/) for more descriptions on how to fit and make predictions using AMPL.

### Pipeline parameters
AMPL includes many parameters to run various model fitting and prediction tasks.
- Pipeline options (parameters) can be set within JSON files containing a parameter list.
- The parameter list with detailed explanations of each option can be found at [atomsci/ddm/docs/PARAMETERS.md](atomsci/ddm/docs/PARAMETERS.md).
- Example pipeline JSON files can be found in the tests directory and the example directory.

### Library documentation
AMPL includes detailed docstrings and comments to explain the modules. Full HTML documentation of the Python library is available with the package at [https://ampl.readthedocs.io/en/latest/](https://ampl.readthedocs.io/en/latest/).

### More information on AMPL usage
- More information on AMPL usage can be found in [Advanced AMPL usage](#advanced-ampl-usage)

---
## Tests
AMPL includes a suite of software tests. This section explains how to run a very simple test that is fast to run. The Python test fits a random forest model using Mordred descriptors on a set of compounds from Delaney, *et al* with solubility data. A molecular scaffold-based split is used to create the training and test sets. In addition, an external holdout set is used to demonstrate how to make predictions on new compounds.

To run the Delaney Python script that curates a dataset, fits a model, and makes predictions, run the following commands:
```
source $ENVROOT/atomsci-env/bin/activate # activate your pip environment.
cd atomsci/ddm/test/integrative/delaney_RF
pytest
```
> ***Note***: *This test generally takes a few minutes on a modern system*

The important files for this test are listed below:

- `test_delany_RF.py`: This script loads and curates the dataset, generates a model pipeline object, and fits a model. The model is reloaded from the filesystem and then used to predict solubilities for a new dataset.
- `config_delaney_fit_RF.json`: Basic parameter file for fitting
- `config_delaney_predict_RF.json`: Basic parameter file for predicting

### More example and test information
- More details on examples and tests can be found in [Advanced testing](#advanced-testing).

---
## Advanced AMPL usage

### Command line
AMPL can **fit** models from the command line with:
```bash
python model_pipeline.py --config_file filename.json # [filename].json is the name of the config file
```

To get more info on an AMPL config file, please refer to:

  - [AMPL Features](https://github.com/ATOMScience-org/AMPL#ampl-features)
  - [Running AMPL](https://github.com/ATOMScience-org/AMPL#running-ampl)
  - [AMPL Tutorials](atomsci/ddm/examples/tutorials)

### Hyperparameter optimization
<details><summary>Hyperparameter optimization for AMPL model fitting is available to run on SLURM clusters or with [Optuna](https://optuna.readthedocs.io/) (Bayesian Optimization). To run Bayesian Optimization, the following steps can be followed.</summary>

1. (Optional) Install Optuna with "pip install optuna"
2. Pre-split your dataset with computed_descriptors if you want to use Mordred/MOE/RDKit descriptors.
3. In the config JSON file, set the following parameters.

   - "hyperparam": "True"
   - "search_type": "optuna"
   - "descriptor_type": "mordred_filtered,rdkit_raw" (use comma to separate multiple values)
   - "model_type": "RF|20" (the number after | is the number of evaluations of Bayesian Optimization)
   - "featurizer": "ecfp,computed_descriptors" (use comma if you want to try multiple featurizers, note the RF and graphconv are not compatible)
   - "result_dir": "/path/to/save/the/final/results,/temp/path/to/save/models/during/optimization" (Two paths separated by a comma)

   RF model specific parameters:
   - "rfe": "uniformint|8,512", (RF number of estimators)
   - "rfd": "uniformint|8,512", (RF max depth of the decision tree)
   - "rff": "uniformint|8,200", (RF max number of features)

    Use the following schemes to define the searching domains

    method|parameter1,parameter2...

    method: supported searching schemes in Optuna include: choice, uniform, loguniform, uniformint. For details, see the [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html).

    parameters:
      - choice: all values to search from, separated by comma, e.g. choice|0.0001,0.0005,0.0002,0.001
      - uniform: low and high bound of the interval to search, e.g. uniform|0.00001,0.001
      - loguniform: low and high bound (in natural log) of the interval to search, e.g. loguniform|-13.8,-6.9
        - **Note**: For backwards compatibility, loguniform values are submitted to Optuna in log scale. Although Optuna supports non-log-scaled value ranges for log-uniform distributions, we maintain the original log-scaled specification format to ensure consistency with existing configurations.
      - uniformint: low and high bound of the interval as integers, e.g. uniformint|8,256

    NN model specific parameters:
     - "lr": "loguniform|-13.8,-6.9", (learning rate)
     - "ls": "uniformint|3|8,512", (layer_sizes)
        - The number between two bars (|) is the number of layers, namely 3 layers, each one with 8~512 nodes
        - Note that the number of layers (number between two |) can not be changed during optimization, if you want to try different number of layers, just run several optimizations.
     - "dp": "uniform|3|0,0.4", (dropouts)
        - 3 layers, each one has a dropout range from 0 to 0.4
        - Note that the number of layers (number between two |) can not be changed during optimization, if you want to try different number of layers, just run several optimizations.

    XGBoost model specific parameters:
     - "xgbg": "uniform|0,0.4", (xgb_gamma, Minimum loss reduction required to make a further partition on a leaf node of the tree)
     - "xgbl": "loguniform|-6.9,-2.3", (xgb_learning_rate, Boosting learning rate (xgboost's "eta"))

4. Run hyperparameter search in batch mode or submit a slurm job.

    ```
    python hyperparam_search_wrapper.py --config_file filename.json
    ```

5. Save a checkpoint to continue it later.

    To save a checkpoint file of the hyperparameter search job, you want to set the following two parameters.
    - "hp_checkpoint_save": "/path/to/the/checkpoint/file.pkl"
    - "hp_checkpoint_load": "/path/to/the/checkpoint/file.pkl"

    If the "hp_checkpoint_load" is provided, the hyperparameter search will continue from the checkpoint.
</details>

---

## Advanced testing
### Running all tests
To run the full set of tests, use Pytest from the test directory:
```bash
source $ENVROOT/atomsci-env/bin/activate # activate your pip environment. "atomsci" is an example here.
cd atomsci/ddm/test
pytest
```

### Running SLURM tests
<details><summary>Several of the tests take some time to fit. These tests can be submitted to a SLURM cluster as a batch job.</summary> Example general SLURM submit scripts are included as `pytest_slurm.sh`.

```bash
source $ENVROOT/atomsci-env/bin/activate # activate your pip environment. "atomsci-env" is an example here.
cd atomsci/ddm/test/integrative/delaney_NN
sbatch pytest_slurm.sh
cd ../../../..
cd atomsci/ddm/test/integrative/wenzel_NN
sbatch pytest_slurm.sh
```
</details>

### Running tests without internet access
<details><summary>AMPL works without internet access. Curation, fitting, and prediction do not require internet access.</summary>

However, the public datasets used in tests and examples are not included in the repo due to licensing concerns. These are automatically downloaded when the tests are run.

If a system does not have internet access, the datasets will need to be downloaded before running the tests and examples. From a system with internet access, run the following shell script to download the public datasets. Then, copy the AMPL directory to the offline system.

```
cd atomsci/ddm/test
bash download_datset.sh
cd ../../..
# Copy AMPL directory to offline system
```
</details>

---
## AMPL tutorials
Please follow link, ["atomsci/ddm/examples/tutorials"](https://github.com/ATOMScience-org/AMPL/tree/master/atomsci/ddm/examples/tutorials), to access a collection of AMPL tutorial notebooks. The tutorial notebooks give an exhaustive coverage of AMPL features. The AMPL team has prepared the tutorials to help beginners understand the basics to advanced AMPL features, and a reference for advanced AMPL users.

---
## Development
### Installing the AMPL for development
Using "pip install -e ." will create a namespace package in your environment directory that points back to your git working directory, so every time you reimport a module you'll be in sync with your working code. Since site-packages is already in your sys.path, you won't have to fuss with PYTHONPATH or setting sys.path in your notebooks.

### Code Push Policy
It's recommended to use a development branch to do the work. After each release, there will be a branch opened for development.

The policy is

1. Create a branch based off a development ("1.6.0 "for example) or "master" branch
2. Create a pull request. Assign a reviewer to approve the code changes

> ***Note***:
> Step 2 is required for pushing directly to "master". For a development branch, this step is recommended but not required.

### Docstring format
The ["Google docstring"](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings) format is used in the AMPL code. When writing new code, please use the same Docstring style. Refer [here](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google) and [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for examples.

### Versioning
Versions are managed through GitHub tags on this repository.

### Built with
- [DeepChem](https://github.com/deepchem/deepchem): A rich repository of chemistry-specific model types and utilities
- [RDKit](https://github.com/rdkit/rdkit): Molecular informatics library
- [Mordred](https://github.com/mordred-descriptor/mordred): Chemical descriptors
- Other Python package dependencies

---
## Project information
### Authors
**[The Accelerating Therapeutics for Opportunities in Medicine (ATOM) Consortium](https://atomscience.org)**

- Amanda J. Minnich <sub>(1)</sub>
- Kevin McLoughlin <sub>(1)</sub>
- Margaret Tse <sub>(2)</sub>
- Jason Deng <sub>(2)</sub>
- Andrew Weber <sub>(2)</sub>
- Neha Murad <sub>(2)</sub>
- Benjamin D. Madej <sub>(3)</sub>
- Bharath Ramsundar <sub>(4)</sub>
- Tom Rush <sub>(2)</sub>
- Stacie Calad-Thomson <sub>(2)</sub>
- Jim Brase <sub>(1)</sub>
- Jonathan E. Allen <sub>(1)</sub>
&nbsp;

### Contributors
- [Amanda Paulson](https://github.com/paulsonak) <sub>(5)</sub>
- Stewart He <sub>(1)</sub>
- Da Shi <sub>(6)</sub>
- Ravichandran Sarangan <sub>(7)</sub>
- Jessica Mauvais <sub>(1)</sub>

<sub>1. [Lawrence Livermore National Laboratory](https://www.llnl.gov/)</sub>\
<sub>2. [GlaxoSmithKline Inc.](https://www.gsk.com/en-gb)</sub>\
<sub>3. [Frederick National Laboratory for Cancer Research](https://frederick.cancer.gov)</sub>\
<sub>4. Computable</sub>\
<sub>5. [University of California, San Francisco](https://www.ucsf.edu/)</sub>\
<sub>6. [Schrodinger](https://www.schrodinger.com/)</sub>\
<sub>7. [Leidos](https://www.leidos.com)</sub>
&nbsp;

### Support, Suggestions or Report Issues
- If you have suggestions or like to report issues, please click [here](https://github.com/ATOMScience-org/AMPL/issues).
&nbsp;

### Contributing
Thank you for contributing to AMPL!

- Contributions must be submitted through pull requests.
- All new contributions must adhere to the MIT license.
&nbsp;

### Release
AMPL is distributed under the terms of the MIT license. All new contributions must be made under this license.

See [MIT license](LICENSE) and [NOTICE](NOTICE) for more details.

- LLNL-CODE-795635
- CRADA TC02264
