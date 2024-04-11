import atomsci.ddm.pipeline.compare_models as cm
import atomsci.ddm.pipeline.parameter_parser as pp
from atomsci.ddm.pipeline.compare_models import nan
import sys
import os
import shutil
import tarfile
import json
import glob
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../delaney_Panel'))
from test_delaney_panel import init, train_and_predict

sys.path.append(os.path.join(os.path.dirname(__file__), '../dc_models'))
from test_retrain_dc_models import H1_curate
from atomsci.ddm.utils import llnl_utils

def clean():
    delaney_files = glob.glob('delaney-processed*.csv')
    for df in delaney_files:
        if os.path.exists(df):
            os.remove(df)

    h1_files = glob.glob('H1_*.csv')
    for hf in h1_files:
        if os.path.exists(hf):
            os.remove(hf)

    if os.path.exists('result'):
        shutil.rmtree('result')

    if os.path.exists('scaled_descriptors'):
        shutil.rmtree('scaled_descriptors')

def get_tar_metadata(model_tarball):
    tarf_content = tarfile.open(model_tarball, "r")
    metadata_file = tarf_content.getmember("./model_metadata.json")
    ext_metadata = tarf_content.extractfile(metadata_file)

    meta_json = json.load(ext_metadata)
    ext_metadata.close()

    return meta_json

def confirm_perf_table(json_f, df):
    """df should contain one entry for the model specified by json_f

    checks to see if the parameters extracted match what's in config
    """
    # should only have trained one model
    assert len(df) == 1
    # the one row
    row = df.iloc[0]

    with open(json_f) as f:
        config = json.load(f)

    model_type = config['model_type']
    if model_type == 'NN':
        assert row['best_epoch'] >= 0
        assert row['max_epochs'] == int(config['max_epochs'])
        assert row['learning_rate'] == float(config['learning_rate'])
        assert row['layer_sizes'] == config['layer_sizes']
        assert row['dropouts'] == config['dropouts']
    elif model_type == 'RF':
        print(row[[c for c in df.columns if c.startswith('rf_')]])
        assert row['rf_estimators'] == int(config['rf_estimators'])
        assert row['rf_max_features'] == int(config['rf_max_features'])
        assert row['rf_max_depth'] == int(config['rf_max_depth'])
    elif model_type == 'xgboost':
        print(row[[c for c in df.columns if c.startswith('xgb_')]])
        assert row['xgb_gamma'] == float(config['xgb_gamma'])
        assert row['xgb_learning_rate'] == float(config['xgb_learning_rate'])
    else:
        assert model_type in pp.model_wl
        assert row['best_epoch'] >= 0
        pparams = pp.wrapper(config)
        assert row['learning_rate'] == float(pparams.learning_rate)

def compare_dictionaries(ref, model_info):
    """Args:
        ref: this is the hardcoded reference dictionary. Everything in this
            dictionary must appear in output and they must be exactly the same or,
            if it's a numeric value, must be within 1e-6.

        model_info: This is the output from get_bset_perf_table

    Returns:
        None
    """
    for k, v in ref.items():
        if not v  == v:
            # in the case of nan
            assert not model_info[k] == model_info[k]
        elif v is None:
            assert model_info[k] is None
        elif type(v) == str:
            assert model_info[k] == v
        else:
            # some kind of numerical object
            assert abs(model_info[k]-v) < 1e-6

def all_similar_tests(json_f, prefix='delaney-processed'):
    train_and_predict(json_f, prefix=prefix)

    df1 = cm.get_filesystem_perf_results('result', 'regression')
    confirm_perf_table(json_f, df1)

    df2 = cm.get_summary_perf_tables(result_dir='result', prediction_type='regression')
    confirm_perf_table(json_f, df2)

    model_uuid = df2['model_uuid'].values[0]
    model_info = cm.get_best_perf_table(metric_type='r2_score', model_uuid=model_uuid, result_dir='result')
    print('model_info:', model_info)
    confirm_perf_table(json_f, pd.DataFrame([model_info]))

    assert model_info['model_parameters_dict'] == df1.iloc[0]['model_parameters_dict']
    assert model_info['model_parameters_dict'] == df2.iloc[0]['model_parameters_dict']

    return df1, df2, model_info

def test_RF_results():
    clean()
    init()
    json_f = 'jsons/reg_config_delaney_fit_RF_mordred_filtered.json'

    df1, df2, model_info = all_similar_tests(json_f)

    # here's a hard coded result to compare to. Things that change run to run have been deleted
    ref = {'collection_name': None, 'model_type': 'RF', 'featurizer': 'computed_descriptors', 
    'splitter': 'scaffold',
    'bucket': 'public', 'descriptor_type': 'mordred_filtered', 'num_samples': nan, 
    'rf_estimators': 501, 'rf_max_features': 33, 'rf_max_depth': 10000, 'max_epochs': nan,
    'best_epoch': nan, 'learning_rate': nan, 'layer_sizes': nan, 'dropouts': nan, 'xgb_gamma': nan, 
    'xgb_learning_rate': nan}

    compare_dictionaries(ref=ref, model_info=model_info)

    assert json.loads(model_info['model_parameters_dict']) == {'rf_estimators':501, 
            'rf_max_features':33,
            'rf_max_depth':10000}

    assert model_info['feat_parameters_dict'] == json.dumps({})

    clean()

def test_NN_results():
    clean()
    init()
    json_f = 'jsons/reg_config_delaney_fit_NN_graphconv.json'

    df1, df2, model_info = all_similar_tests(json_f)

    # don't compare best_epoch
    model_params = json.loads(model_info['model_parameters_dict'])
    del model_params['best_epoch']
    assert model_params == {"dropouts": [0.10,0.10,0.10],
            "layer_sizes": [64,64,64],
            "learning_rate": 0.000753,
            "max_epochs": 5,}

    assert model_info['feat_parameters_dict'] == json.dumps({})

    clean()

def test_XGB_results():
    clean()
    init()
    json_f = 'jsons/reg_config_delaney_fit_XGB_mordred_filtered.json'

    df1, df2, model_info = all_similar_tests(json_f)

    assert json.loads(model_info['model_parameters_dict']) == {'xgb_colsample_bytree': 1.0,
            'xgb_gamma': 0.1, 'xgb_learning_rate': 0.11, 'xgb_max_depth': 6,
            'xgb_min_child_weight': 1.0, 'xgb_n_estimators': 100, 'xgb_subsample': 1.0}

    assert model_info['feat_parameters_dict'] == json.dumps({})

    clean()

def test_AttentiveFP_results():
    clean()
    H1_curate()
    json_f = 'jsons/reg_config_H1_fit_AttentiveFPModel.json'

    df1, df2, model_info = all_similar_tests(json_f, 'H1')

    # don't compare best_epoch
    model_params = json.loads(model_info['model_parameters_dict'])
    del model_params['best_epoch']
    assert model_params == {
        "max_epochs": 5,
        "AttentiveFPModel_mode":"regression",
        "AttentiveFPModel_num_layers":3,
        "AttentiveFPModel_learning_rate": 0.0007,
        "AttentiveFPModel_model_dir": "result",
        "AttentiveFPModel_n_tasks": 1,}

    assert json.loads(model_info['feat_parameters_dict']) == {"MolGraphConvFeaturizer_use_edges":"True",}

    clean()

def test_GCN_results():
    clean()
    H1_curate()
    json_f = 'jsons/reg_config_H1_fit_GCNModel.json'

    df1, df2, model_info = all_similar_tests(json_f, 'H1')

    # don't compare best_epoch
    model_params = json.loads(model_info['model_parameters_dict'])
    del model_params['best_epoch']
    assert model_params == {
        "max_epochs": 5,
        "GCNModel_n_tasks": 1,
        "GCNModel_model_dir": "result",
        "GCNModel_mode": "regression",
        "GCNModel_learning_rate": 0.0003,
        "GCNModel_graph_conv_layers": [16,16],
        "GCNModel_predictor_hidden_feats": 16,}

    assert model_info['feat_parameters_dict'] == json.dumps({})

    clean()

def test_GraphConvModel_results():
    clean()
    H1_curate()
    json_f = 'jsons/reg_config_H1_fit_GraphConvModel.json'

    df1, df2, model_info = all_similar_tests(json_f, 'H1')

    # don't compare best_epoch
    model_params = json.loads(model_info['model_parameters_dict'])
    del model_params['best_epoch']
    assert model_params == {
        "max_epochs": 5,
        "GraphConvModel_n_tasks": 1,
        "GraphConvModel_model_dir": "result",
        "GraphConvModel_mode": "regression",
        "GraphConvModel_dropout": 0.2,
        "GraphConvModel_learning_rate": 0.000753,
        "GraphConvModel_graph_conv_layers": [64,64,64],
        "GraphConvModel_dense_layer_size": 64,}

    assert model_info['feat_parameters_dict'] == json.dumps({})

    clean()

def test_MPNN_results():
    if not llnl_utils.is_lc_system():
        assert True
        return

    clean()
    H1_curate()
    json_f = 'jsons/reg_config_H1_fit_MPNNModel.json'

    df1, df2, model_info = all_similar_tests(json_f, 'H1')

    # don't compare best_epoch
    model_params = json.loads(model_info['model_parameters_dict'])
    del model_params['best_epoch']
    assert model_params == {
        "max_epochs": 5,
        "MPNNModel_n_tasks": 1,
        "MPNNModel_mode": "regression",
        "MPNNModel_model_dir": "result",
        "MPNNModel_learning_rate": 0.0005,
        "MPNNModel_n_atom_feat": 75,
        "MPNNModel_n_pair_feat": 14,}

    assert model_info['feat_parameters_dict'] == json.dumps({})

    clean()

def test_PytorchMPNN_results():
    clean()
    H1_curate()
    json_f = 'jsons/reg_config_H1_fit_PytorchMPNNModel.json'

    df1, df2, model_info = all_similar_tests(json_f, 'H1')

    model_params = json.loads(model_info['model_parameters_dict'])
    del model_params['best_epoch']
    ref = {
        "max_epochs": 5,
        "PytorchMPNNModel_model_dir": "result",
        "PytorchMPNNModel_mode": "regression",
        "PytorchMPNNModel_learning_rate": 0.001,
        "PytorchMPNNModel_n_tasks": 1,}
    assert model_params == ref
    assert json.loads(model_info['feat_parameters_dict']) == {"MolGraphConvFeaturizer_use_edges":"True"}

    clean()

if __name__ == '__main__':
    test_RF_results()
    test_NN_results()
    test_XGB_results()
    test_AttentiveFP_results()
    test_GCN_results()
    test_GraphConvModel_results()
    test_MPNN_results()
    test_PytorchMPNN_results()
