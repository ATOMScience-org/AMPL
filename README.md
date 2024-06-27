[![License](http://img.shields.io/:license-mit-blue.svg)](https://github.com/ATOMScience-org/AMPL/blob/master/LICENSE) [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/atomscience) 

| [Install](#install) | [Docker](#install-with-docker) | [Tutorials](#ampl-tutorials) |  [Features](#ampl-features) | [Pipeline parameters](atomsci/ddm/docs/PARAMETERS.md) | [Docs](https://ampl.readthedocs.io/en/latest/) |

# ATOM Modeling PipeLine (AMPL) for Drug Discovery
An open-source, end-to-end software pipeline for data curation, model building, and molecular property prediction to advance in silico drug discovery.

*Created by the [Accelerating Therapeutics for Opportunities in Medicine (ATOM) Consortium](https://atomscience.org)*

<img src="atomsci/ddm/docs/ATOM_cymatics_black_wordmark.jpg" width="370" height="100" class="center"></img>


![Static Badge](https://img.shields.io/badge/Announcement-1.6.1-blue)![Static Badge](https://img.shields.io/badge/Announcement-1.6.1-blue)![Static Badge](https://img.shields.io/badge/Announcement-1.6.1-blue)![Static Badge](https://img.shields.io/badge/Announcement-1.6.1-blue)![Static Badge](https://img.shields.io/badge/Announcement-1.6.1-blue)![Static Badge](https://img.shields.io/badge/Announcement-1.6.1-blue)

## Check out our new tutorial series that walks through AMPL's end-to-end modeling pipeline to build a machine learning model! View them in our [docs](https://ampl.readthedocs.io/en/latest/) or as Jupyter notebooks in our [repo](https://github.com/ATOMScience-org/AMPL/tree/master/atomsci/ddm/examples/tutorials).

The ATOM Modeling PipeLine (AMPL) extends the functionality of DeepChem and supports an array of machine learning and molecular featurization tools to predict key potency, safety and pharmacokinetic-relevant parameters. AMPL has been benchmarked on a large collection of pharmaceutical datasets covering a wide range of parameters. This is a living software project with active development. Check back for continued updates. Feedback is welcomed and appreciated, and the project is open to contributions! An [article describing the AMPL project](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b01053) was published in JCIM. For those without access to JCIM, a preprint of the article is available on [ArXiv](http://arxiv.org/abs/1911.05211). [Documentation is available here.](https://ampl.readthedocs.io/en/latest/pipeline.html)


---
## Table of contents
- [Install](#install)
   - [Quick Install](#installation-quick-summary)
   - [Jupyter kernel](#create-jupyter-notebook-kernel-optional)
   - [Docker](#install-with-docker)
   - [Uninstall](#uninstall)
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
## Install
AMPL 1.6 supports Python 3.9 CPU or CUDA-enabled machines using CUDA 11.8 on Linux. All other systems are experimental. For a quick install summary, see [here](#install-summary). We do not support other CUDA versions because there are multiple ML package dependency conflicts that can occur. For more information you can look at [DeepChem](https://deepchem.readthedocs.io/en/latest/get_started/installation.html), [TensorFlow](https://www.tensorflow.org/install/pip), [PyTorch](https://pytorch.org/get-started/locally/), [DGL](https://www.dgl.ai/pages/start.html) or [Jax](https://github.com/google/jax#installation).

### Create pip environment

#### 1. Create a virtual env with Python 3.9 
Make sure to create your virtual env in a convenient directory that has at least 12Gb space.

Go to the directory where the new environment directory be installed in. Define an environment variable - "ENVROOT".

```bash
export ENVROOT=~/workspace # for LLNL LC users, use your workspace
or
export ENVROOT=~ # or the directory as your environment root
cd $ENVROOT
```

> *We use "workspace" and "atomsci-env" as an example here.*

```bash
# LLNL only: 
# module load python/3.9.12
cd $ENVROOT
python3.9 -m venv atomsci-env
```

#### 2. Activate the environment
```bash
source $ENVROOT/atomsci-env/bin/activate
```

#### 3. Update pip
```bash
pip install pip --upgrade
```

#### 4. Clone AMPL repository
```bash
git clone https://github.com/ATOMScience-org/AMPL.git 
```

#### 5. Install pip requirements
Depending on system performance, creating the environment can take some time.
> ***Note:*** *Based on which environment (CPU or CUDA) to run on, only run one of the following:*

- CPU-only installation:
```bash
cd AMPL/pip
pip install -r cpu_requirements.txt
```

- CUDA installation:

First load the CUDA module. Then run cuda specific package install.

```bash
cd AMPL/pip
# LLNL only: 
# module load cuda/11.8
pip install -r cuda_requirements.txt
```
If you get `out of memory` errors, try setting these environment variables:
```
export LD_LIBRARY_PATH=<your_env>/lib:$LD_LIBRARY_PATH
export PYTHONUSERBASE=<your_env>
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=48
export PYTORCH_HIP_ALLOC_CONF=gargage_collection_threshold:0.9,max_split_size_mb:128
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### 6. *(Optional) LLNL LC only*: if you use [model_tracker](https://ampl.readthedocs.io/en/latest/pipeline.html#module-pipeline.model_tracker), install atomsci.clients
```bash
# LLNL only: required for ATOM model_tracker
pip install -r clients_requirements.txt
```

### Install AMPL
Run the following to build the "atomsci" modules. This is required.

```bash
# return to AMPL parent directory
cd ..
./build.sh
pip install -e .
```
---
## Installation Quick Summary
```bash
export ENVROOT=~/workspace           # set ENVROOT example
cd $ENVROOT                          # go to a convenient home directory
# LLNL only:
# module load python/3.9.12

python3.9 -m venv atomsci-env        # create environment with Python 3.9
source $ENVROOT/atomsci-env/bin/activate 
pip install pip --upgrade               

git clone https://github.com/ATOMScience-org/AMPL.git # clone AMPL
cd AMPL/pip    
# LLNL only:
# If use CUDA: 
# module load cuda/11.8                         
pip install -r cpu_requirements.txt    # install cpu_requirements.txt OR cuda_requirements.txt  

# LLNL only: required for ATOM model_tracker
# pip install -r clients_requirements.txt

cd ..                                   
./build.sh                            
pip install -e .                        
```
---
## Create jupyter notebook kernel (optional)
To run AMPL from Jupyter Notebook, with your environment activated. To setup a new kernel:

```
python -m ipykernel install --user --name atomsci-env
```
---
## Install with Docker
- Download and install Docker Desktop.
  - https://www.docker.com/get-started
- Create a workspace folder to mount with Docker environment and transfer files. 
- Get the Docker image and run it.
  ```
  docker pull atomsci/atomsci-ampl
  docker run -it -p 8888:8888 -v </local_workspace_folder>:</directory_in_docker> atomsci/atomsci-ampl
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

> ***Note***: *Be sure to save any work you want to be permanent in your workspace folder. If the container is shut down, you'll lose anything not in that folder.*  
---

## Uninstall
To remove AMPL from a pip environment use:
```bash
pip uninstall atomsci-ampl
```

To remove an entire virtual environment named "atomsci-env":
```bash
rm -rf $ENVROOT/atomsci-env
```

To remove cached packages and clear space:
```bash
pip cache purge
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
- Hyperparameter optimization  

### 4. Supported models
- scikit-learn random forest models
- XGBoost models
- Fully connected neural networks
- Graph convolution models  

### 5. Visualization and analysis
- Visualization and analysis tools  
</details>
Details of running specific features are within the [parameter (options) documentation](#Pipeline-parameters). More detailed documentation is in the [library documentation](#Library-documentation).  

---
## Running AMPL
AMPL can be run from the command line or by importing into Python scripts and Jupyter notebooks.  

### Python scripts and Jupyter notebooks
AMPL can be used to fit and predict molecular activities and properties by importing the appropriate modules. See the [examples](atomsci/ddm/examples/tutorials/14_BSEP_modeling.ipynb) for more descriptions on how to fit and make predictions using AMPL.

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
<details><summary>Hyperparameter optimization for AMPL model fitting is available to run on SLURM clusters or with [HyperOpt](https://hyperopt.github.io/hyperopt/) (Bayesian Optimization). To run Bayesian Optimization, the following steps can be followed.</summary>

1. (Optional) Install HyperOpt with "pip install hyperopt"
2. Pre-split your dataset with computed_descriptors if you want to use Mordred/MOE/RDKit descriptors.
3. In the config JSON file, set the following parameters.
   
   - "hyperparam": "True"
   - "search_type": "hyperopt"
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
    
    method: supported searching schemes in HyperOpt include: choice, uniform, loguniform, uniformint, see https://github.com/hyperopt/hyperopt/wiki/FMin for details.
    
    parameters:
      - choice: all values to search from, separated by comma, e.g. choice|0.0001,0.0005,0.0002,0.001
      - uniform: low and high bound of the interval to serach, e.g. uniform|0.00001,0.001
      - loguniform: low and high bound (in natural log) of the interval to serach, e.g. uniform|-13.8,-6.9
      - uniformint: low and high bound of the interval as integers, e.g. uniforming|8,256
  
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
