# ATOM Modeling PipeLine (AMPL) for Drug Discovery

*Created by the [Accelerating Therapeutics for Opportunites in Medicine (ATOM) Consortium](https://atomscience.org)*

<img src="atomsci/ddm/docs/ATOM_cymatics_black_wordmark.jpg" width="370" height="100" class="center"></img>


AMPL is an open-source, modular, extensible software pipeline for building and sharing models to advance in silico drug discovery.

> The ATOM Modeling PipeLine (AMPL) extends the functionality of DeepChem and supports an array of machine learning and molecular featurization tools. AMPL is an end-to-end data-driven modeling pipeline to generate machine learning models that can predict key safety and pharmacokinetic-relevant parameters. AMPL has been benchmarked on a large collection of pharmaceutical datasets covering a wide range of parameters.

A pre-print of a manuscript describing this project will be available through [ArXiv](https://arxiv.org/).  
&nbsp;  

## Public beta release
**This release marks the first public availability of the ATOM Modeling PipeLine (AMPL). Installation instructions for setting up and running AMPL are described below. Basic examples of model fitting and prediction are also included. AMPL has been deployed to and tested in multiple computing environments by ATOM Consortium members. Detailed documentation for the majority of the available features is included, but the documentation does not cover *all* developed features. This is a living software project with active development. Check back for continued updates. Feedback is welcomed and appreciated, and the project is open to contributions!**  
&nbsp;  

---
## Table of contents
- [Getting started](#Getting-started)
  - [Prerequisites](#Prerequisites)
  - [Install](#Install)
- [Examples and tests](#Examples-and-tests)
- [Features](#Features)
- [Running AMPL](#Running-AMPL)
- [Advanced AMPL usage](#Advanced-AMPL-usage)
- [Advanced installation](#Advanced-installation)
- [Advanced testing](#Advanced-testing)
- [Development](#Development)
- [Project information](#Project-information)  
&nbsp;  

## Useful links
- [Pipeline parameters (options)](atomsci/ddm/docs/PARAMETERS.md)
- [Library documentation](atomsci/ddm/docs/build/html/index.html)  
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

./build.sh && ./install.sh
```

- After this process, you will have an `atomsci` conda environment with all dependencies installed. The name of the AMPL package is `atomsci-ampl` and is installed in the `install.sh` script to the environment with conda's `pip`.
&nbsp;  

#### More installation information
- More details on installation can be found in [Advanced installation](#Advanced-installation).  
&nbsp;  
&nbsp;  


---
<a name="Examples-and-tests"></a>
## Examples and tests
AMPL includes a suite of examples and tests for demonstration and testing. One test fits a random forest model using Mordred descriptors on a set of compounds from Delaney, *et al* with solubility data. A molecular scaffold-based split is used to create the training and test sets. In addition, an external holdout set is used to demonstrate how to make predictions on new compounds.

To run an example Python script that curates a dataset, fits a model, and makes predictions, run the following commands:
```
conda activate atomsci

cd atomsci/ddm/test/integrative/delaney_RF

pytest
```
&nbsp;  
&nbsp;  
- *Note: This test generally takes a few minutes on a modern system*

The important files for running this test are listed below:

- `test_delany_RF.py`: This script loads and curates the dataset, generates a model pipeline object, and fits a model. The model is reloaded from the filesystem and then used to predict solubilities for a new dataset.
- `config_delaney_fit_RF.json`: Basic parameter file for fitting
- `config_delaney_predict_RF.json`: Basic parameter file for predicting

- *Note: Further pipeline examples can be found in the [`atomsci/ddm/test/integrative`](atomsci/ddm/test/integrative) directory.*  
&nbsp;  

### More example and test information
- More details on examples and tests can be found in [Advanced testing](#Advanced-testing).
&nbsp;  
&nbsp;  


---
<a name="Features"></a>
## Features
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

#### 4. Supported models
- scikit-learn random forest models
- XGBoost models
- Fully connected neural networks
- Graph convolution models  
&nbsp;  

### 5. Visualization and analysis
- Visualization and analysis tools  
&nbsp;  
&nbsp;  


---
<a name="Running-AMPL"></a>
## Running AMPL
AMPL can be run from the command line or by importing into Python scripts and Jupyter notebooks.  
&nbsp;  

### Python scripts and Jupyter notebooks
AMPL can be used to fit and predict molecular activities and properties by importing the appropriate modules. See the examples for more descriptions on how to fit and make predictions using AMPL.  
&nbsp;  

### Pipeline parameters (options)
AMPL includes many parameters to run various model fitting and prediction tasks.
- Pipeline options (parameters) can be set within JSON files containing a parameter list.
- The parameter list with detailed explanations of each option can be found at `atomsci/ddm/docs/PARAMETERS.md`.
- Example pipeline JSON files can be found in the tests directory and the example directory.  
&nbsp;  

### Library documentation
AMPL includes detailed docstrings and comments to explain the modules. Full HTML documentation of the Python library is available with the package at `atomsci/ddm/docs/build/html/index.html`.  
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
Hyperparameter optimization for AMPL model fitting is available to run on SLURM clusters. Examples of running hyperparameter optimization will be added.  
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
- Jonathan E. Allen (1)
- Margaret Tse (2)
- Jason Deng (2)
- Andrew Weber (2)
- Neha Murad (2)
- Benjamin D. Madej (3)
&nbsp;  

1. Lawrence Livermore National Laboratory
2. GlaxoSmithKline Inc.
3. Frederick National Laboratory for Cancer Research  
&nbsp;  

### Support
Please contact the AMPL repository owners for bug reports, questions, and comments.  
&nbsp;  

### Contributing
Thank you for contributing to AMPL!

- Contributions must be submitted through pull requests. Please let the repository owners know about new pull requests.
- All new contributions must be made under the MIT license.  
&nbsp;  

### Release
AMPL is distributed under the terms of the MIT license. All new contributions must be made under this license.

See [MIT license](mit.txt) and [NOTICE](NOTICE) for more details.

- LLNL-CODE-795635
- CRADA TC02264
&nbsp;  
&nbsp;  

### Readme date
November 7, 2019  
&nbsp;  
&nbsp;  
