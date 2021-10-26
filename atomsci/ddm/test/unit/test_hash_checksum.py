import os

import atomsci.ddm.utils.checksum_utils as cu

def clean():
    pass

def test_create_checksum():
    csv_file = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')
    hash_value = '491463a16315d70ee973c9eb699f36f9'

    assert cu.create_checksum(csv_file) == hash_value

def test_uses_same_training_data_by_datasets():
    file1 = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')
    file2 = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')

    assert cu.uses_same_training_data_by_datasets(file1, file2) == True

def test_uses_same_training_data_not_equals_by_datasets():
    file1 = os.path.abspath('../../examples/tutorials/datasets/HTR3A_ChEMBL.csv')
    file2 = os.path.abspath('../../examples/tutorials/datasets/DTC_HTR3A.csv')

    assert cu.uses_same_training_data_by_datasets(file1, file2) == False


def test_uses_same_training_data_by_tars():
    tar1 = os.path.abspath('../integrative/hyperopt/tmp/H1_std_model_f131896c-388b-48c7-b60a-046e82375b09.tar.gz')
    tar2 = os.path.abspath('../integrative/hyperopt/tmp/H1_std_model_f131896c-388b-48c7-b60a-046e82375b09.tar.gz')

    assert cu.uses_same_training_data_by_tarballs(tar1, tar2) == True

def test_uses_same_training_data_not_equals_by_tars():
    tar1 = os.path.abspath('../integrative/delaney_NN/result/delaney-processed_curated_fit_model_01f75c05-f859-4b31-ac6c-0010781ef0df.tar.gz')
    tar2 = os.path.abspath('../integrative/hyperopt/tmp/H1_std_model_f131896c-388b-48c7-b60a-046e82375b09.tar.gz')

    assert cu.uses_same_training_data_by_tarballs(tar1, tar2) == False

if __name__ == '__main__':
    test()