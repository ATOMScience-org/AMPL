### BSEP datasets for training and testing machine learning models with AMPL

This directory contains datasets that can be used to train and test BSEP inhibition models using the
ATOM Modeling PipeLine (AMPL). The files are as follows:

- `morgan_warner_combined_bsep_data.csv`: This is the dataset used to train, validate and evaluate the open data models described in
the paper. It contains data selected from two publications , as described in the Methods section of the paper:
 - [Morgan et al. (2013)](http://dx.doi.org/10.1093/toxsci/kft176)
 - [Warner et al. (2012)](http://dx.doi.org/10.1124/dmd.112.047068)

- `ChEMBL25_BSEP_curated_data.csv`: This is a dataset produced by querying IC50 data from the 
[ChEMBL 25 database](https://www.ebi.ac.uk/chembl/) for human ABCB11 (target ID CHEMBL6020). The data were filtered to 
remove outliers and restricted to assays that measured [3H]-taurocholate transport into inverted cell membrane vesicles.
pIC50 values were computed and averaged for compounds with multiple observations, using the MLE procedure described in the paper.
Note that this dataset includes most of the data from the Morgan (2013) and Warner datasets.

- `ChEMBL25_BSEP_test_data.csv`: Subset of `ChEMBL25_BSEP_curated_data.csv` that excludes compounds appearing in the Morgan/Warner combined dataset.

- `small_test_data.csv`: A small subset of `ChEMBL25_BSEP_test_data.csv` to be used for running quick tests of the prediction code.

Data from ChEMBL are provided under a Creative Commons Attribution-ShareAlike 3.0 Unported license. See the file LICENSE.pdf in the
parent of this directory for the terms of the license.
