import os
import glob

import atomsci.ddm.utils.checksum_utils as cu
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp

import logging
log = logging.getLogger(__name__)

def clean():
    """
    Clean test files
    """
    if "result" not in os.listdir():
        os.mkdir("result")
    for f in os.listdir("./result"):
        if os.path.isfile("./result/"+f):
            os.remove("./result/"+f)

def test_create_checksum():
    csv_file = os.path.abspath('../../../examples/archive/tutorials2021/datasets/HTR3A_ChEMBL.csv')
    hash_value = '491463a16315d70ee973c9eb699f36f9'

    assert cu.create_checksum(csv_file) == hash_value

def test_uses_same_training_data_by_datasets():
    file1 = os.path.abspath('../../../examples/archive/tutorials2021/datasets/HTR3A_ChEMBL.csv')
    file2 = os.path.abspath('../../../examples/archive/tutorials2021/datasets/HTR3A_ChEMBL.csv')

    assert cu.uses_same_training_data_by_datasets(file1, file2) == True

def test_uses_same_training_data_not_equals_by_datasets():
    file1 = os.path.abspath('../../../examples/archive/tutorials2021/datasets/HTR3A_ChEMBL.csv')
    file2 = os.path.abspath('../../../examples/archive/tutorials2021/datasets/DTC_HTR3A.csv')

    assert cu.uses_same_training_data_by_datasets(file1, file2) == False

# train the same file twice. then compare the checksums. they should match
def test_uses_same_training_data_by_tars_delaney_train_NN():
    # test 1
    json_file = 'jsons/config_delaney_train_NN.json'
    ds_key_file = '../../test_datasets/delaney-processed_curated_fit.csv'

    tar1 = train_and_get_tar(json_file, ds_key_file)
    tar2 = train_and_get_tar(json_file, ds_key_file)

    assert cu.uses_same_training_data_by_tarballs(tar1, tar2) == True

# train the same file twice. then compare the checksums. they should match
def test_uses_same_training_data_by_tars_nn_ecfp():
    # test 2
    json_file = 'jsons/nn_ecfp.json'
    ds_key_file = '../../test_datasets/aurka_chembl_base_smiles_union.csv'

    tar1 = train_and_get_tar(json_file, ds_key_file)
    tar2 = train_and_get_tar(json_file, ds_key_file)

    assert cu.uses_same_training_data_by_tarballs(tar1, tar2) == True

# train the same file twice. then compare the checksums. they should match
def test_uses_same_training_data_by_tars_delaney_train_RF_mordred():
    # test 3
    json_file = 'jsons/reg_config_delaney_fit_RF_mordred_filtered.json'
    ds_key_file = '../../test_datasets/delaney-processed_curated_fit.csv'

    tar1 = train_and_get_tar(json_file, ds_key_file)
    tar2 = train_and_get_tar(json_file, ds_key_file)

    assert cu.uses_same_training_data_by_tarballs(tar1, tar2) == True

def test_uses_same_training_data_not_equals_by_tars():
    json_file1 = 'jsons/config_delaney_train_NN.json'
    json_file2 = 'jsons/nn_ecfp.json'
    ds_key_file1 = '../../test_datasets/delaney-processed_curated_fit.csv'
    ds_key_file2 = '../../test_datasets/aurka_chembl_base_smiles_union.csv'

    tar1 = train_and_get_tar(json_file1, ds_key_file1)
    tar2 = train_and_get_tar(json_file2, ds_key_file2)

    assert cu.uses_same_training_data_by_tarballs(tar1, tar2) == False

def train_and_get_tar(input_json, ds_key_file):

    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, input_json)    

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path,  ds_key_file)
    pparams.result_dir = os.path.join(script_path, 'result')

    train_pipe = mp.ModelPipeline(pparams)
    train_pipe.train_model()

    list_of_files = glob.glob('./result/*.gz') # check all *.gz
    latest_file = max(list_of_files, key=os.path.getctime) # get the latest gz

    return latest_file

if __name__ == '__main__':
    # Clean
    # -----
    clean()
    
    test()