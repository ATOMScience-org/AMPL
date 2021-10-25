import os

import atomsci.ddm.utils.checksum_utils as cu

def clean():
    pass

def test_create_checksum():
    csv_file = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')
    hash_value = '491463a16315d70ee973c9eb699f36f9'

    assert cu.create_checksum(csv_file) == hash_value

def test_checksum_equals():
    file1 = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')
    file2 = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')

    assert cu.checksum_matches(file1, file2) == True

def test_checksum_not_equals():
    file1 = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')
    file2 = os.path.abspath('../../examples/tutorials/datasets/DTC_HTR3A.csv')

    assert cu.checksum_matches(file1, file2) == False

def test_uses_same_training_data_wo_hash():
    file1 = os.path.abspath('../integrative/delaney_NN/result/delaney-processed_curated_fit_model_01f75c05-f859-4b31-ac6c-0010781ef0df.tar.gz')
    file2 = os.path.abspath('../integrative/hyperopt/tmp/H1_std_model_0bad6950-19d5-4a84-812f-11f90878e169.tar.gz')

    assert cu.uses_same_training_data(file1, file2) == False

def test_uses_same_training_data_w_hash():
    file1 = os.path.abspath('/g/g20/mauvais2/atom/mauvais2/model_retrain/cyp3a4_union_trainset_base_smiles_model_e1ab0834-7247-4bd7-99df-5fd434eed11f.tar.gz')
    file2 = os.path.abspath('/g/g20/mauvais2/atom/mauvais2/model_retrain/cyp3a4_union_trainset_base_smiles_model_b54a4f38-95e2-4cf4-907b-d52a5b22e91e.tar.gz')

    assert cu.uses_same_training_data(file1, file2) == True

if __name__ == '__main__':
    test()