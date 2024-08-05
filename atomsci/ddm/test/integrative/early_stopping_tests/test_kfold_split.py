import copy
import pandas as pd
import numpy as np
import json
import os

import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.predict_from_model as pfm
from atomsci.ddm.utils import llnl_utils

import sklearn.metrics as skmetrics

def get_test_set(dskey, split_csv, id_col):
    """using model_metadata read dataset key and split uuid to split dataset into
    component parts

    Parameters:
        dskey: path to csv file
        split_csv: path to split csv

    Returns:
        train_df, valid_df, test_df
    """
    df = pd.read_csv(dskey)
    split_df = pd.read_csv(split_csv)

    test_df = df[df[id_col].isin(split_df[split_df['subset']=='test']['cmpd_id'])]

    return test_df

def split(pparams):
    split_params = copy.copy(pparams)
    split_params.split_only=True
    split_params.previously_split=False

    model_pipeline = mp.ModelPipeline(split_params)
    # comment out this line after splitting once so you don't re-split
    split_uuid = model_pipeline.split_dataset()

    return split_uuid

def train(pparams):
    train_pipe = mp.ModelPipeline(pparams)
    train_pipe.train_model()

    return train_pipe

def find_best_test_metric(model_metrics):
    for metric in model_metrics:
        if metric['label']=='best' and metric['subset']=='test':
            return metric

    return None

def test_kfold():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'nn_ecfp_kfold.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path

    saved_model_identity(pparams)

def test_graphconv():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'nn_graph.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path
    pparams.split_uuid = 'test-split'

    saved_model_identity(pparams)

def test_ecfp_nn():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'nn_ecfp.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path
    pparams.split_uuid = 'test-split'

    saved_model_identity(pparams)

def test_train_valid_test():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'nn_ecfp_random.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path

    saved_model_identity(pparams)

def test_attentivefp():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'attentivefp_random.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path

    saved_model_identity(pparams)

def test_gcnmodel():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'gcnmodel_random.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path

    saved_model_identity(pparams)

def test_graphconvmodel():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'GraphConvModel_random.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path

    saved_model_identity(pparams)

def test_mpnnmodel():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'MPNNModel_random.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path

    saved_model_identity(pparams)

def test_pytorchmpnnmodel():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'PytorchMPNNModel_random.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key = os.path.join(script_path, 
        '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir = script_path

    saved_model_identity(pparams)

def saved_model_identity(pparams):
    if not llnl_utils.is_lc_system():
        assert True
        return
        
    script_path = os.path.dirname(os.path.realpath(__file__))
    if not pparams.previously_split:
        split_uuid = split(pparams)

        pparams.split_uuid = split_uuid
        pparams.previously_split = True

    train_pipe = train(pparams)
    split_csv = os.path.join(script_path, '../../test_datasets/', 
                    train_pipe.data._get_split_key())
    test_df = get_test_set(pparams.dataset_key, split_csv, pparams.id_col)

    # verify
    with open(os.path.join(pparams.output_dir, 'model_metrics.json'), 'r') as f:
        model_metrics = json.load(f)

    # only compare test results since there's only one fold for test
    metrics = find_best_test_metric(model_metrics)
    id_col = metrics['input_dataset']['id_col']
    response_col = metrics['input_dataset']['response_cols'][0]
    smiles_col = metrics['input_dataset']['smiles_col']
    test_length = metrics['prediction_results']['num_compounds']

    model_tar = train_pipe.params.model_tarball_path
    pred_df = pfm.predict_from_model_file(model_tar, test_df, id_col=id_col,
                smiles_col=smiles_col, response_col=response_col)
    pred_df2 = pfm.predict_from_model_file(model_tar, test_df, id_col=id_col,
                smiles_col=smiles_col, response_col=response_col)

    X = pred_df[response_col+'_actual'].values
    y = pred_df[response_col+'_pred'].values
    X2 = pred_df2[response_col+'_actual'].values
    y2 = pred_df2[response_col+'_pred'].values

    r2 = skmetrics.r2_score(X, y)
    rms = np.sqrt(skmetrics.mean_squared_error(X, y))
    mae = skmetrics.mean_absolute_error(X, y)

    saved_r2 = metrics['prediction_results']['r2_score']
    saved_rms = metrics['prediction_results']['rms_score']
    saved_mae = metrics['prediction_results']['mae_score']

    print(metrics['subset'])
    print(pred_df.columns)
    print(abs(r2-saved_r2))
    print(abs(rms-saved_rms))
    print(abs(mae-saved_mae))
    print(np.mean(abs(y2-y)))

    assert abs(r2-saved_r2)<1e-5 \
            and abs(rms-saved_rms)<1e-5 \
            and abs(mae-saved_mae)<1e-5 \
            and np.mean(abs(y2-y))<1e-5 \
            and (test_length == len(test_df))

if __name__ == '__main__':
    print('train_valid_test')
    test_train_valid_test()
    print('kfold')
    test_kfold()
    print('graphconv')
    test_graphconv()
    print('ecfp nn')
    test_ecfp_nn()
    print('attentive fp')
    test_attentivefp()
    print('gcn model')
    test_gcnmodel()
    print('graphconv new')
    test_graphconvmodel()
    print('mpnn')
    test_mpnnmodel()
    print('pytorch mpnn')
    test_pytorchmpnnmodel()
    print('Passed')

