## Example models and scripts for BSEP inhibition prediction

This directory contains example models and code for the following paper:
Kevin S. McLoughlin, Claire G. Jeong, Thomas D. Sweitzer, Amanda J. Minnich,
Margaret J. Tse, Brian J. Bennion, Jonathan E. Allen, Stacie Calad-Thomson, Thomas S. Rush, and James M. Brase.
"Machine Learning Models to Predict Inhibition of the Bile Salt Export Pump".

A preprint of the paper is available at [J Chem Inf Model](https://pubmed.ncbi.nlm.nih.gov/33502191/).

To run the models:

- Clone the AMPL git repository and install the software, as described 
in [the README file for AMPL](https://github.com/ATOMScience-org/AMPL/blob/master/README.md).
- Prepare your input data file. It should be in CSV format with a row of column headers at the top, and should (at minimum)
contain a column of SMILES strings for the compounds you want to run predictions on. You may optionally include the following columns:
  - A column of unique compound IDs; if none is provided, one will be generated for you. 
  - A column of binary activity values, if you have measured IC50s for the input compounds: 0 if the IC50 < 100 uM, 1 if IC50 >= 100 uM.
- From the Unix command line, cd to this directory (`...AMPL/atomsci/ddm/examples/BSEP`) and run the command:
`./predict_bsep_inhibition.py --help`
to see the full set of command line options. You need to specify the `--input_file` and `--output_file` options at minimum. The other options
specify the SMILES, compound ID and activity columns in the input file, determine how the input SMILES strings are processed, and
indicate the specific model that is used for predictions.

For example, the following command runs predictions on the small test dataset included in the `data` subdirectory:

`./predict_bsep_inhibition.py -i data/small_test_data.csv -o small_test_output.csv --id_col compound_name --smiles_col base_rdkit_smiles --activity_col active`

When the `--activity_col` option is specified, it is asssumed that the input activity values are the ground truth. The code then compares the input values
against the predictions and computes and displays various performance metrics.

The accessibility domain index can optionally be calculated by including a path to the original data used to train the model. If the training data is not found at its original location or at the path included, the AD index will not be calculated. The following command will run predictions and calculate the AD index:

`./predict_bsep_inhibition.py -i data/small_test_data.csv -o small_test_output.csv --id_col compound_name --smiles_col base_rdkit_smiles --activity_col active --ad_method z_score --ext_train_data data/morgan_warner_combined_bsep_data.csv`

### Output file format
The output of the `predict_bsep_inhibition.py` command is a CSV file with the following columns (not necessarily in this order):
- Compound ID
- SMILES strings as input
- Standardized SMILES strings (unless the `--dont_standardize` option is specified)
- The input activity values, if the `--activity_col` option is specified, in column `<activity_col>_actual`
- The predicted probabilities of each compound to be a BSEP inhibitor, in column `active_prob` or `(activity_col)_prob`
- The predicted activity value for each compound (1 if a BSEP inhibitor, 0 if not), in column `active_pred` or `(activity_col)_pred`

### Error handling
The software uses RDKit to process the input SMILES strings and the MolVS package to standardize them and remove salts (unless the 
`dont_standardize` option is selected). If any of the input SMILES strings cannot be parsed by RDKit, they are excluded from the dataset
and predictions are run on the remaining compounds. The program will print warning messages naming the SMILES strings it was unable
to process. Most other warning messages may be ignored.

### Expected run time
Run times are mainly driven by the time it takes to compute Mordred descriptors for each compound, which scales with the number of 
compounds. On our development Linux system, the average run time is 9 + 0.586N seconds where N is the number of compounds.

### License
Models and data under this directory are provided under a Creative Commons Attribution-ShareAlike 3.0 Unported license. See the file
LICENSE.pdf in this directory for the terms of the license.
