#!/usr/bin/env python

import os
import subprocess

filename = 'small_test_output.csv'

# do not include the original training data, and verify that still able to predict with the model but can not calculate the ADI
def test_without_adi():
    print("\nWithout ADI")
    result = subprocess.Popen(['python', '../../../examples/BSEP/predict_bsep_inhibition.py', '-i',  '../../../examples/BSEP/data/small_test_data.csv', '-o', 'small_test_output.csv', '--id_col', 'compound_name', '--smiles_col', 'base_rdkit_smiles', '--activity_col', 'active'])

    output, error = result.communicate()
    assert (error is None) # should have no error

    if os.path.exists(filename):
        os.remove(filename)

# do include the original training data, and verify that can predict and also calculate the ADI
def test_with_adi():
    print("\nWith ADI")
    
    result = subprocess.Popen(['python', '../../../examples/BSEP/predict_bsep_inhibition.py', '-i',  '../../../examples/BSEP/data/small_test_data.csv', '-o', 'small_test_output.csv', '--id_col', 'compound_name', '--smiles_col', 'base_rdkit_smiles', '--activity_col', 'active', '--ad_method', 'z_score', '--ext_train_data', '../../../examples/BSEP/data/data/morgan_warner_combined_bsep_data.csv'])
    output, error = result.communicate()
    assert (error is  None) # should have no error

    if os.path.exists(filename):
        os.remove(filename)

if __name__ == '__main__':
    test_without_adi()
    test_with_adi()
