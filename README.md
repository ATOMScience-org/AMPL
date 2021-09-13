# ATOM Modeling PipeLine (AMPL) for Drug Discovery
[![License](http://img.shields.io/:license-mit-blue.svg)](https://github.com/ATOMconsortium/AMPL/blob/master/LICENSE)

*Created by the [Accelerating Therapeutics for Opportunites in Medicine (ATOM) Consortium](https://atomscience.org)*

<img src="atomsci/ddm/docs/ATOM_wordmark_black_transparent.png" width="370" height="100" class="center"></img>


AMPL is an open-source, modular, extensible software pipeline for building and sharing models to advance in silico drug discovery.

> The ATOM Modeling PipeLine (AMPL) extends the functionality of DeepChem and supports an array of machine learning and molecular featurization tools. AMPL is an end-to-end data-driven modeling pipeline to generate machine learning models that can predict key safety and pharmacokinetic-relevant parameters. AMPL has been benchmarked on a large collection of pharmaceutical datasets covering a wide range of parameters.

An [article describing the AMPL project](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b01053) 
was published in JCIM. For those without access to JCIM, a preprint of the article is available on 
[ArXiv](http://arxiv.org/abs/1911.05211). 

Documentation in readthedocs format is available [here](https://ampl.readthedocs.io/en/latest/pipeline.html).
&nbsp;  

## Public release
**This release marks the first public availability of the ATOM Modeling PipeLine (AMPL). Installation instructions for setting up and running AMPL are described below. Basic examples of model fitting and prediction are also included. AMPL has been deployed to and tested in multiple computing environments by ATOM Consortium members. Detailed documentation for the majority of the available features is included, but the documentation does not cover *all* developed features. This is a living software project with active development. Check back for continued updates. Feedback is welcomed and appreciated, and the project is open to contributions!**  
&nbsp;  

---
## Table of contents
- [Getting started](#Getting-started)
  - [Prerequisites](#Prerequisites)
  - [Install](#Install)
  - [Install with Docker](#Install-docker)
  - [AMPL usage survey](#AMPL-usage-survey)
- [Tutorials](#AMPL-tutorials)
- [Tests](#Tests)
- [AMPL Features](#AMPL-Features)
- [Running AMPL](#Running-AMPL)
- [Advanced AMPL usage](#Advanced-AMPL-usage)
- [Advanced installation](#Advanced-installation)
- [Advanced testing](#Advanced-testing)
- [Development](#Development)
- [Project information](#Project-information)  
&nbsp;  

## Useful links
- [Pipeline parameters (options)](atomsci/ddm/docs/PARAMETERS.md)
- [Library documentation](https://ampl.readthedocs.io/en/latest/index.html)  
&nbsp;  
&nbsp;  


---
<a name="Getting-started"></a>
## Getting started
Welcome to the ATOM Modeling PipeLine (AMPL) for Drug Discovery! These instructions will explain how to install this pipeline for model fitting and prediction.  
&nbsp;  

<a name="Prerequisites"></a>
### Prerequisites
AMPL is a Python 3 package that has been developed and run in a specific conda environment. The following prerequisites are necessary to install AMPL:

- conda (Anaconda 3 or Miniconda 3, Python 3)  
&nbsp;  

<a name="Install"></a>
### Install
#### Clone the git repository

`git clone https://github.com/ATOMconsortium/AMPL.git`  
&nbsp;  

#### Create conda environment
```
cd conda

conda create -y -n atomsci --file conda_package_list.txt

conda activate atomsci

pip install -r pip_requirements.txt
```

- *Note: Depending on system performance, creating the environment can take some time.*  
&nbsp;  

#### Install AMPL
Go to the `AMPL` root directory and install the AMPL package:
```
conda activate atomsci

cd ..

./build.sh && ./install.sh system
```

- The `install.sh system` command installs AMPL directly in the conda environment. If `install.sh` alone is used, then AMPL is installed in the `$HOME/.local` directory.

- After this process, you will have an `atomsci` conda environment with all dependencies installed. The name of the AMPL package is `atomsci-ampl` and is installed in the `install.sh` script to the environment with conda's `pip`.
&nbsp;  

#### More installation information
- More details on installation can be found in [Advanced installation](#Advanced-installation).  
&nbsp;  

<a name="Install-docker"></a>
### Install with Docker
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
  - Replace the `d33b0faf6bc9` with `localhost`
  - If this doesn't work, exit the container and change port from 8888 to some other number such as 7777 or 8899 (in all 3 places it's written), then rerun both commands
- Be sure to save any work you want to be permanent in your workspace folder. If the container is shut down, you'll lose anything not in that folder.
&nbsp; 

<a name="AMPL-usage-survey"></a>
### AMPL usage survey 
- The ATOM team would like to hear your experiences in using the AMPL software. The team actively uses feedback to develop the best possible drug discovery modeling pipeline. Click [here](https://docs.google.com/forms/d/e/1FAIpQLSdM7mRXsvRr4MXUIoWgh9hQSmAOFgwXD3nLA4sLpNygec6hzA/viewform?usp=sf_link) to access the survey link. The survey typically takes 5 minutes to complete. Thank you for providing your feedback. 
&nbsp;
&nbsp;

---
<a name="AMPL-tutorials"></a>
## AMPL tutorials
 Please follow the link, [`atomsci/ddm/examples/tutorials`](https://github.com/ATOMconsortium/AMPL/tree/master/atomsci/ddm/examples/tutorials), to access a collection of AMPL tutorial COLAB (Jupyter) notebooks. The tutorial notebooks give an exhaustive coverage of AMPL features. The AMPL team has prepared the tutorials to help beginners understand the basics to advanced AMPL features, and a reference for advanced AMPL users. 
&nbsp;  
&nbsp;  


---
<a name="Tests"></a>
## Tests
AMPL includes a suite of software tests. This section explains how to run a very simple test that is fast to run. The Python test fits a random forest model using Mordred descriptors on a set of compounds from Delaney, *et al* with solubility data. A molecular scaffold-based split is used to create the training and test sets. In addition, an external holdout set is used to demonstrate how to make predictions on new compounds.

To run the Delaney Python script that curates a dataset, fits a model, and makes predictions, run the following commands:
```
conda activate atomsci

cd atomsci/ddm/test/integrative/delaney_RF

pytest
```
- *Note: This test generally takes a few minutes on a modern system*  
&nbsp;  
&nbsp;  

The important files for this test are listed below:

- `test_delany_RF.py`: This script loads and curates the dataset, generates a model pipeline object, and fits a model. The model is reloaded from the filesystem and then used to predict solubilities for a new dataset.
- `config_delaney_fit_RF.json`: Basic parameter file for fitting
- `config_delaney_predict_RF.json`: Basic parameter file for predicting
&nbsp;  

### More example and test information
- More details on examples and tests can be found in [Advanced testing](#Advanced-testing).
&nbsp;  
&nbsp;  


---
<a name="AMPL-Features"></a>
## AMPL Features
AMPL enables tasks for modeling and prediction from data ingestion to data analysis and can be broken down into the following stages:

1. Data ingestion and curation
2. Featurization
3. Model training and tuning
4. Prediction generation
5. Visualization and analysis  
&nbsp;  

### 1. Data curation
- Generation of RDKit molecular SMILES structures
- Processing of qualified or censored data processing
- Curation of activity and property values
&nbsp;  

### 2. Featurization
- Extended connectivity fingerprints (ECFP)
- Graph convolution latent vectors from DeepChem
- Chemical descriptors from Mordred package
- Descriptors generated by MOE (requires MOE license)  
&nbsp;  

### 3. Model training and tuning
- Test set selection
- Cross-validation
- Uncertainty quantification
- Hyperparameter optimization  
&nbsp;  

### 4. Supported models
- scikit-learn random forest models
- XGBoost models
- Fully connected neural networks
- Graph convolution models  
&nbsp;  

### 5. Visualization and analysis
- Visualization and analysis tools  
&nbsp;  

Details of running specific features are within the [parameter (options) documentation](#Pipeline-parameters). More detailed documentation is in the [library documentation](#Library-documentation).  
&nbsp;  
&nbsp;  


---
<a name="Running-AMPL"></a>
## Running AMPL
AMPL can be run from the command line or by importing into Python scripts and Jupyter notebooks.  
&nbsp;  

### Python scripts and Jupyter notebooks
AMPL can be used to fit and predict molecular activities and properties by importing the appropriate modules. See the [examples](#Example-AMPL-usage) for more descriptions on how to fit and make predictions using AMPL.  
&nbsp;  

<a name="Pipeline-parameters"></a>
### Pipeline parameters (options)
AMPL includes many parameters to run various model fitting and prediction tasks.
- Pipeline options (parameters) can be set within JSON files containing a parameter list.
- The parameter list with detailed explanations of each option can be found at [atomsci/ddm/docs/PARAMETERS.md](atomsci/ddm/docs/PARAMETERS.md).
- Example pipeline JSON files can be found in the tests directory and the example directory.  
&nbsp;  

<a name="Library-documentation"></a>
### Library documentation
AMPL includes detailed docstrings and comments to explain the modules. Full HTML documentation of the Python library is available with the package at [atomsci/ddm/docs/build/html/index.html](atomsci/ddm/docs/build/html/index.html).  
&nbsp;  

### More information on AMPL usage
- More information on AMPL usage can be found in [Advanced AMPL usage](#Advanced-AMPL-usage)  
&nbsp;  
&nbsp;  


---
<a name="Advanced-AMPL-usage"></a>
## Advanced AMPL usage

### Command line
AMPL can **fit** models from the command line with:
```
python model_pipeline.py --config_file test.json
```  
&nbsp;  

### Hyperparameter optimization
Hyperparameter optimization for AMPL model fitting is available to run on SLURM clusters or with [HyperOpt](https://hyperopt.github.io/hyperopt/) (Bayesian Optimization). To run Bayesian Optimization, the following steps can be followed.

1. (Optional) Install HyperOpt with `pip install hyperopt`
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

&nbsp;  
&nbsp;  


---
<a name="Advanced-installation"></a>
## Advanced installation
### Deployment
AMPL has been developed and tested on the following Linux systems:

- Red Hat Enterprise Linux 7 with SLURM
- Ubuntu 16.04  
&nbsp;  

### Uninstallation
To remove AMPL from a conda environment use:
```
conda activate atomsci
pip uninstall atomsci-ampl
```
&nbsp;  

To remove the atomsci conda environment entirely from a system use:
```
conda deactivate
conda remove --name atomsci --all
```
&nbsp;  
&nbsp;  


---
<a name="Advanced-testing"></a>
## Advanced testing
### Running all tests
To run the full set of tests, use Pytest from the test directory:
```
conda activate atomsci

cd atomsci/ddm/test

pytest
```
&nbsp;  

### Running SLURM tests
Several of the tests take some time to fit. These tests can be submitted to a SLURM cluster as a batch job. Example general SLURM submit scripts are included as `pytest_slurm.sh`.

```
conda activate atomsci

cd atomsci/ddm/test/integrative/delaney_NN

sbatch pytest_slurm.sh

cd ../../../..

cd atomsci/ddm/test/integrative/wenzel_NN

sbatch pytest_slurm.sh
```
&nbsp;  

### Running tests without internet access
AMPL works without internet access. Curation, fitting, and prediction do not require internet access.

However, the public datasets used in tests and examples are not included in the repo due to licensing concerns. These are automatically downloaded when the tests are run. 

If a system does not have internet access, the datasets will need to be downloaded before running the tests and examples. From a system with internet access, run the following shell script to download the public datasets. Then, copy the AMPL directory to the offline system.
```
cd atomsci/ddm/test

bash download_datset.sh

cd ../../..

# Copy AMPL directory to offline system
```
&nbsp;  
&nbsp;  


---
<a name="Development"></a>
## Development
### Installing the AMPL for development
To install the AMPL for development, use the following commands instead:
```
conda activate atomsci
./build.sh && ./install_dev.sh
```
&nbsp;  

This will create a namespace package in your conda  directory that points back to your git working directory, so every time you reimport a module you'll be in sync with your working code. Since site-packages is already in your sys.path, you won't have to fuss with PYTHONPATH or setting sys.path in your notebooks.  
&nbsp;  

### Versioning
Versions are managed through GitHub tags on this repository.  
&nbsp;  

### Built with
- [DeepChem](https://github.com/deepchem/deepchem): The basis for the graph convolution models
- [RDKit](https://github.com/rdkit/rdkit): Molecular informatics library
- [Mordred](https://github.com/mordred-descriptor/mordred): Chemical descriptors
- Other Python package dependencies  
&nbsp;  
&nbsp;  


---
<a name="Project-information"></a>
## Project information
### Authors
**[The Accelerating Therapeutics for Opportunities in Medicine (ATOM) Consortium](https://atomscience.org)**

- Amanda J. Minnich (1)
- Kevin McLoughlin (1)
- Margaret Tse (2)
- Jason Deng (2)
- Andrew Weber (2)
- Neha Murad (2)
- Benjamin D. Madej (3)
- Bharath Ramsundar (4)
- Tom Rush (2)
- Stacie Calad-Thomson (2)
- Jim Brase (1)
- Jonathan E. Allen (1)
&nbsp;  

1. Lawrence Livermore National Laboratory
2. GlaxoSmithKline Inc.
3. Frederick National Laboratory for Cancer Research
4. Computable  
&nbsp;  

### Support
Please contact the AMPL repository owners for bug reports, questions, and comments.  
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
&nbsp;  
&nbsp;  
