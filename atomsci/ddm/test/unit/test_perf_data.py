import atomsci.ddm.pipeline.perf_data as perf_data
import atomsci.ddm.pipeline.model_pipeline as model_pipeline
import atomsci.ddm.pipeline.parameter_parser as parse
import os
import tempfile
import deepchem as dc
import numpy as np
import shutil
import pandas as pd
import json

def copy_to_temp(dskey, res_dir):
    """
    Copy a dataset to a temporary directory.

    Parameters:
    dskey (str): Path to the original dataset.
    res_dir (str): Path to the temporary directory.

    Returns:
    str: Path to the copied dataset in the temporary directory.
    """
    new_dskey = shutil.copy(dskey, res_dir)
    return new_dskey

def setup_paths():
    """
    Set up the paths for the test, including creating a temporary directory and copying the dataset to it.

    Returns:
    tuple: A tuple containing:
        - res_dir (str): Path to the temporary result directory.
        - tmp_dskey (str): Path to the copied dataset in the temporary directory.
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    res_dir = tempfile.mkdtemp()
    dskey = os.path.join(script_path, '../test_datasets/aurka_chembl_base_smiles_union.csv')
    tmp_dskey = copy_to_temp(dskey, res_dir)

    return res_dir, tmp_dskey

def read_params(json_file, res_dir, tmp_dskey):
    """
    Read parameters from a JSON file and update them with the result directory and dataset key.

    Parameters:
    json_file (str): Path to the JSON file containing parameters.
    res_dir (str): Path to the result directory.
    tmp_dskey (str): Path to the copied dataset in the temporary directory.

    Returns:
    dict: Updated parameters.
    """

    with open(json_file, 'r') as file:
        params = json.load(file)
    params['result_dir'] = res_dir
    params['dataset_key'] = tmp_dskey
    return params

def make_relative_to_file(relative_path):
    """
    Generates the full path relative to the location of this file.

    Parameters:
    relative_path (str): The relative path to convert.

    Returns:
    str: The absolute path corresponding to the relative path.
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    result = os.path.join(script_path, relative_path)

    return result

def test_KFoldRegressionPerfData():
    """
    Test the KFoldRegressionPerfData class to ensure it correctly handles k-fold regression performance data.

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(make_relative_to_file('config_perf_data_KFoldRegressoinPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.KFoldRegressionPerfData)

    ids = sorted(list(mp.data.combined_training_data().ids))
    weights = perf.get_weights(ids)
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))

    pred_vals = d.y
    # This should have r2 of 1
    r2 = perf.accumulate_preds(pred_vals, ids)
    assert r2 == 1
    # do a few more folds
    r2 = perf.accumulate_preds(pred_vals, ids)
    r2 = perf.accumulate_preds(pred_vals, ids)

    (res_ids, res_vals, res_std) = perf.get_pred_values()
    (r2_mean, r2_std) = perf.compute_perf_metrics()

    assert np.allclose(res_vals, real_vals)
    assert np.allclose(res_std, np.zeros_like(res_std))

    # perfect score every time
    assert r2_mean==1
    assert r2_std==0

def test_KFoldRegressionPerfDataMulti():
    """
    Test the KFoldRegressionPerfData class for multi-fold regression performance data.
    """
    res_dir, tmp_dskey = setup_paths()

    # duplicate pIC50 column
    df = pd.read_csv(tmp_dskey)
    df['pIC50_dupe'] = df['pIC50']
    df.to_csv(tmp_dskey, index=False)

    params = read_params(make_relative_to_file('config_perf_data_KFoldRegressoinPerfDataMulti.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.KFoldRegressionPerfData)

    ids = sorted(list(mp.data.combined_training_data().ids))
    weights = perf.get_weights(ids)
    assert weights.shape == (len(ids),2)
    assert np.allclose(weights, np.ones_like(weights))

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones_like(weights))

    pred_vals = d.y
    # This should have r2 of 1
    r2 = perf.accumulate_preds(pred_vals, ids)
    assert r2 == 1
    # do a few more folds
    r2 = perf.accumulate_preds(pred_vals, ids)
    r2 = perf.accumulate_preds(pred_vals, ids)

    (res_ids, res_vals, res_std) = perf.get_pred_values()
    (r2_mean, r2_std) = perf.compute_perf_metrics()

    assert np.allclose(res_vals, real_vals)
    assert np.allclose(res_std, np.zeros_like(res_std))

    # perfect score every time
    assert r2_mean==1
    assert r2_std==0

def test_KFoldClassificationPerfData():
    """
    Test the KFoldClassificationPerfData functionality.

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(
        make_relative_to_file('config_perf_data_KFoldClassificationPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.KFoldClassificationPerfData)

    ids = sorted(list(mp.data.combined_training_data().ids))
    weights = perf.get_weights(ids)
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))

    num_classes = 2
    # input to to_one_hot needs to have the shape (N,) not (N,1)
    pred_vals = dc.metrics.to_one_hot(d.y.reshape(len(d.y)), num_classes)
    # This should have r2 of 1
    roc_auc_score = perf.accumulate_preds(pred_vals, ids)
    assert roc_auc_score == 1
    # do a few more folds
    roc_auc_score = perf.accumulate_preds(pred_vals, ids)
    roc_auc_score = perf.accumulate_preds(pred_vals, ids)

    (res_ids, res_classes, res_probs, res_std) = perf.get_pred_values()
    (roc_auc_mean, roc_auc_std) = perf.compute_perf_metrics()

    # std should be zero
    assert all((res_std==np.zeros_like(res_std)).flatten())
    # probs should match predictions
    assert all((res_probs==pred_vals.reshape(len(d.y), 1, num_classes)).flatten())
    # all predictions are correct
    assert all(res_classes==real_vals)
    # perfect score every time
    assert roc_auc_mean==1
    assert roc_auc_std==0

def test_SimpleRegressionPerfData():
    """
    Test the SimpleRegressionPerfData class for correct performance data creation and metrics computation.

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(
        make_relative_to_file('config_perf_data_SimpleRegressionPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.SimpleRegressionPerfData)

    real_vals = perf.get_real_values()
    weights = perf.get_weights()
    ids = np.array(range(len(real_vals))) # these are not used by SimpleRegressionPerfData
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, 
                             ids=ids, w=np.ones(len(ids)))

    pred_vals = d.y
    # This should have r2 of 1 ids are ignored
    r2 = perf.accumulate_preds(pred_vals, ids)
    assert r2 == 1

    (res_ids, res_vals, _) = perf.get_pred_values()
    (r2_mean, _) = perf.compute_perf_metrics()

    # the predicted values should equal the real values
    assert all(real_vals == res_vals)

    # should be a perfect score
    assert r2_mean == 1

def test_SimpleClassificationPerfData():
    """
    Test function for SimpleClassificationPerfData.

    This function sets up a model pipeline, trains a model, and creates performance data
    for a simple classification task. It then verifies the following:

    """
    res_dir, tmp_dskey = setup_paths()

    params = read_params(
        make_relative_to_file('config_perf_data_SimpleClassificationPerfData.json'),
        res_dir, tmp_dskey)

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, 'train')

    assert isinstance(perf, perf_data.SimpleClassificationPerfData)

    ids = sorted(list(mp.data.train_valid_dsets[0][0].ids))
    weights = perf.get_weights()
    real_vals = perf.get_real_values()
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))

    num_classes = 2
    # input to to_one_hot needs to have the shape (N,) not (N,1)
    pred_vals = dc.metrics.to_one_hot(d.y.reshape(len(d.y)), num_classes)
    # This should have r2 of 1
    roc_auc_score = perf.accumulate_preds(pred_vals, ids=ids)
    assert roc_auc_score == 1

    (res_ids, res_classes, res_probs, _) = perf.get_pred_values()
    (roc_auc_mean, _) = perf.compute_perf_metrics()

    # probs should match predictions
    assert all((res_probs==pred_vals.reshape(len(d.y), 1, num_classes)).flatten())
    # all predictions are correct
    assert all(res_classes==real_vals)
    # perfect score every time
    assert roc_auc_mean==1


if __name__ == "__main__":
    test_KFoldRegressionPerfDataMulti()
    test_KFoldRegressionPerfData()
    test_SimpleClassificationPerfData()
    test_KFoldClassificationPerfData()
    test_SimpleRegressionPerfData()