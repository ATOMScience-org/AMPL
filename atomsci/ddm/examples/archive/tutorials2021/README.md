![Test Image 1](Img/ATOM.PNG)

The ATOM Modeling PipeLine (AMPL; https://github.com/ATOMScience-org/AMPL) is an open-source, modular, extensible software pipeline for building and sharing models to advance in silico drug discovery. To see the list of AMPL parameters, please check this link,  https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/docs/PARAMETERS.md

This page contains a collection of AMPL-COLAB tutorial notebooks. 

```diff
+ Please note that if you have trouble opening up any of the following notebooks, please go to, https://nbviewer.jupyter.org/, and paste the notebook link to view the contents.
```


## 0. Basic Google COLAB Introduction (Works best with Google Chrome)
* [Tutorial-00:](00_BasicCOLAB_Tutorial.ipynb) Basic COLAB tutorial. For all the COLAB tutorials, click on the tutorial link, and then click on "Open in Colab" baner. You can open and run the notebook from the browser. If you want to save your edits to the notebook, you need to save a copy in your Google Drive. Usually, Google COLAB saves the notebook files under the "My Drive > Colab Notebooks" folder

## 1. Data Collection and creating Machine-Learning ready datasets:

The data that we gather for modeling is small-molecule/drug binding data. The following links will introduce some of the concepts and outcome measures related to this topic:
* https://en.wikipedia.org/wiki/IC50
* https://bpspubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1476-5381.2009.00604.x

For the tutorials, we will use the small-molecule binding data obtained from either one of the following resources, ChEMBL (https://www.ebi.ac.uk/chembl/), Drug Target Commons (DTC; https://drugtargetcommons.fimm.fi/) and Excape-DB (https://solr.ideaconsult.net/search/excape/). 

Click [here](supp_md/README.md) to learn about single-target focussed data.


### Explore HTR3A binding data from ExCAPE-DB
* [Tutorial-01:](01_Exploring_Target_Activity_ExcapeDB.ipynb) (**Time: ~ 3 minutes**)
This COLAB notebook will use AMPL for data cleaning, EDA and clustering on ExCAPE-DB (https://solr.ideaconsult.net/search/excape/) data for HTR3A protein 

* [Tutorial-02:](02_Explore_Data_ExcapeDB_curation.ipynb) (**Time: ~ 6 minutes**)
This COLAB notebook will use AMPL for data curation of HTR3A protein data from ExCAPE-DB (https://solr.ideaconsult.net/search/excape/) Data 

### Explore HTR3A binding data from Drug Target Commons database

* [Tutorial-03:](03_Explore_Data_DTC.ipynb) (**Time: ~ 4 minutes**)
This COLAB notebook will use AMPL for Data cleaning, EDA and clustering of HTR3A protein data from Drug Target Commons (DTC)  
* [Tutorial-04:](04_Explore_Data_DTC_Curate.ipynb) (**Time: ~ 10 minutes**)
This COLAB notebook will use AMPL for Data curation of HTR3A protein data from Drug Target Commons (DTC)

### Curating, merging and visualizing two datasets 
* [Tutorial-05:](05_EDA_Curate_Merge_Visualize.ipynb) (**Time: ~ 4 minutes**)
This COLAB notebook will use AMPL to upload datasets (small-molecule activity data from ChEMBL), clean, merge and do some basic Exploratory Data Analysis. 
* [Tutorial-06:](06_Combine_Datasets.ipynb) (**Time: ~ 4 minutes**)
This COLAB notebook with use AMPL to merge HTR3A binding data from two different data sources, DTC and ExCAPE-DB.

### Creating Unionset Train/Test data creation 
* [Tutorial-06a:](06a_UnionDSet.ipynb) (**Time: ~ 6 minutes**)
This COLAB notebook will use AMPL for creating Union train/test datasets. 

### Exploratory Data Analysis (EDA) Notebooks
* [Tutorial-07:](07_EDA_With_Harmonization.ipynb) (**Time: ~ 4 minutes**). The notebook uses HTR3A as the protein target. The notebook accomplishes the following tasks:
   * Uses AMPL software
   * Reads in data from three database sources: ChEMBL, Excape-DB and DTC 
   * Cleans, standardizes and analyzes the data
   * Merges and harmonizes to create a dataset
* [Tutorial-08:](08_AMPL_EDA_Part2.ipynb) Exploratory Data Analysis-Regression 
* [Tutorial-09:](09_AMPL_EDA_Part2_Classification.ipynb) Exploratory Data Analysis-Regression 

## 2. Model training and tuning:

### Random Forest modeling to predict solubility
* [Tutorial-10:](10_Delaney_Solubility_Prediction.ipynb) (**Time: ~ 2 minutes**): Simple supervised learning example.
AMPL will read the public data (117 chemical compounds), curate, fit a Random Forest model to predict solubility and test the model. For additional information on the dataset, please check this publication,https://pubmed.ncbi.nlm.nih.gov/15154768/
![Delaney](https://github.com/ravichas/AMPL-Tutorial/blob/master/Img/Delaney.PNG)

### Graph Convolution modeling to predict SCN5A binding affinities 
* [Tutorial-11:](11_CHEMBL26_SCN5A_IC50_prediction.ipynb) (**Time: ~ 18 minutes**): 
This COLAB notebook will use AMPL for predicting binding affinities -pIC50 values- of ligands that could bind to human **Sodium channel protein type 5 subunit alpha** protein (Gene: SCN5A) using Graph Convolutional Network Model. ChEMBL database is the data source of binding affinities (pIC50)
![Test Image 1](https://github.com/ravichas/AMPL-Tutorial/blob/master/Img/SCN5A.PNG)

## 3. Hyper-parameter Optimization (HPO), Uncertainty Quantification (UQ), and using metrics for analyzing model performance. 

This notebook also explores AMPL functions for saving and loading prebuild AMPL models for analysis. 
* [Tutorial-12](12_AMPL_HPO_demo.ipynb) (**Time: ~ 30 minutes**) Hyper-parameter Optimization [(HPO)](https://en.wikipedia.org/wiki/Hyperparameter_optimization) and Uncertainty Quantification [(UQ)](https://en.wikipedia.org/wiki/Uncertainty_quantification).

## 4. Creating high-quality models 
* Please refer to [08_AMPL_EDA_Part2.ipynb](08_AMPL_EDA_Part2.ipynb) and [09_AMPL_EDA_Part2_Classification.ipynb](09_AMPL_EDA_Part2_Classification.ipynb) that describe helpful hints for visualizing HPO results and methods to identify best performing models.

## 5. Model Inference: 
* [Tutorial-14:](14_BSEP_modeling.ipynb) This notebook creates an AMPL (RF) model using BSEP dataset (reference: https://pubmed.ncbi.nlm.nih.gov/33502191/), and makes predictions (inference) on an external sample test dataset.   
 
## AMPL for drug-discovery Workshops
* ATOM (AMPL)-FNL Workshop-1: (June 05, 2021) <br>
In this workshop, we will demonstrate how to use Atom Modeling PipeLine (AMPL; https://github.com/ATOMScience-org/AMPL), a new, open-source pip-based software that automates key drug discovery steps. The workshop is part of the NCI Data Science Learning Exchange Hands-on Tutorial series. Click [here](https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/examples/tutorials/AMPL_FNL_Workshop_06052021.ipynb) to access the notebook. For use with GCP or other pre-installed AMPL environments, click [here](https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/examples/tutorials/GCP_AMPL_FNL_Workshop_06052021.ipynb).
* ATOM(AMPL)-FNL Workshop-2: (Aug 2021)<br>
The goal of the workshop is to take single target binding assay data collected from three different chemoinformatics data sources (ChEMBL, DTC and ExCAPE-DB) and use AMPL to create a single harmonized Machine-Learning ready data (Aug, 2021)
    - Notebook1: Click [here](https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/examples/tutorials/AMPL_FNL_Wrshp2_1.ipynb) to access the notebook. 
    - Notebook2: Click [here](https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/examples/tutorials/AMPL_FNL_Wrshp2_2.ipynb) to access the notebook.
    - Notebook3: Click [here](https://github.com/ATOMScience-org/AMPL/blob/master/atomsci/ddm/examples/tutorials/AMPL_FNL_Wrshp2_3.ipynb) to access the notebook.

## Supporting links

### Similar chemoinformatics, drug-discovery software tools:
* DeepChem, https://deepchem.io/
* rdkit, https://www.rdkit.org/

### Chemoinformatics databases
* ChEMBL: https://www.ebi.ac.uk/chembl/
* PubChem: https://pubchem.ncbi.nlm.nih.gov/
* Drug Target Commons (DTC): https://drugtargetcommons.fimm.fi/
* ExCAPE-DB: https://solr.ideaconsult.net/search/excape/
* DrugBank: https://go.drugbank.com/

## Acknowledgements
Most of the tutorial code chunks came from multple Jupyter notebooks generously shared by the ATOM team. 
* Amanda Paulson
* Ben Madej 
* Da Shi
* Hiran Ranganathan
* Jessica Mauvais
* Jonathan Allen
* Kevin Mcloughlin
* Sarangan Ravichandran
* Stewart He
* Ya Ju Fan
* Contributions from the following student programs: 
    * The Purdue Data Mine; https://datamine.purdue.edu/  
    * Butler University
    * Columbia University 
