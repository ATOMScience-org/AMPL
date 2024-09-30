import atomsci.ddm.pipeline.perf_data as perf_data
import atomsci.ddm.pipeline.model_pipeline as model_pipeline
import atomsci.ddm.pipeline.parameter_parser as parse
import os
import tempfile
import pdb
import deepchem as dc
import numpy as np

def test_KFoldRegressionPerfData():
    script_path = os.path.dirname(os.path.realpath(__file__))
    res_dir = tempfile.mkdtemp()

    params = {"verbose": "True",
    "datastore": "False", 
    "save_results": "False", 
    "model_type": "NN", 
    "featurizer": "ecfp", 
    "prediction_type": "classification",
    "split_strategy": "k_fold_cv", 
    "splitter": "random",
    "split_test_frac": "0.15", 
    "split_valid_frac": "0.15", 
    "transformers": "True", 
    "dataset_key": os.path.join(script_path,
        '../test_datasets/aurka_chembl_base_smiles_union.csv'),
    "id_col": "compound_id", 
    "response_cols":"pIC50", 
    "smiles_col": "base_rdkit_smiles",
    "max_epochs":"2",
    "prediction_type": "regression",
    "result_dir":res_dir}

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, mp.model_wrapper.transformers, 'train')

    ids = sorted(list(mp.data.train_valid_dsets[0][0].ids[:5]))
    weights = perf.get_weights(ids)
    assert weights.shape == (5,1)
    assert all(weights==1)

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))
    # pass correct values through the transformers
    for t in perf.transformers:
        d = t.transform(d)

    pred_vals = d.y
    # This should have r2 of 1
    r2 = perf.accumulate_preds(pred_vals, ids)
    assert r2 == 1
    # do a few more folds
    r2 = perf.accumulate_preds(pred_vals, ids)
    r2 = perf.accumulate_preds(pred_vals, ids)

    (res_ids, res_vals, res_std) = perf.get_pred_values()
    (r2_mean, r2_std) = perf.compute_perf_metrics()

    print(f"should be {real_vals}, [0,0,0,0,0]")
    print(res_vals, res_std)

def test_KFoldClassificationPerfData():
    script_path = os.path.dirname(os.path.realpath(__file__))
    res_dir = tempfile.mkdtemp()

    params = {"verbose": "True",
    "datastore": "False", 
    "save_results": "False", 
    "model_type": "NN", 
    "featurizer": "ecfp", 
    "prediction_type": "classification",
    "split_strategy": "k_fold_cv", 
    "splitter": "random",
    "split_test_frac": "0.15", 
    "split_valid_frac": "0.15", 
    "transformers": "True", 
    "dataset_key": os.path.join(script_path,
        '../test_datasets/aurka_chembl_base_smiles_union.csv'),
    "id_col": "compound_id", 
    "response_cols":"active", 
    "smiles_col": "base_rdkit_smiles",
    "max_epochs":"2",
    "prediction_type": "classification",
    "result_dir":res_dir}

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, mp.model_wrapper.transformers, 'train')

    ids = sorted(list(mp.data.train_valid_dsets[0][0].ids))
    weights = perf.get_weights(ids)
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    real_vals = perf.get_real_values(ids)
    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))
    # There should be no transformers
    assert len(perf.transformers) == 0

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
    script_path = os.path.dirname(os.path.realpath(__file__))
    res_dir = tempfile.mkdtemp()

    params = {"verbose": "True",
    "datastore": "False", 
    "save_results": "False", 
    "model_type": "NN", 
    "featurizer": "ecfp", 
    "prediction_type": "classification",
    "split_strategy": "train_valid_test", 
    "splitter": "random",
    "split_test_frac": "0.15", 
    "split_valid_frac": "0.15", 
    "transformers": "True", 
    "dataset_key": os.path.join(script_path,
        '../test_datasets/aurka_chembl_base_smiles_union.csv'),
    "id_col": "compound_id", 
    "response_cols":"pIC50", 
    "smiles_col": "base_rdkit_smiles",
    "max_epochs":"2",
    "prediction_type": "regression",
    "result_dir":res_dir}

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, mp.model_wrapper.transformers, 'train')

    real_vals = perf.get_real_values()
    weights = perf.get_weights()
    ids = np.array(range(len(real_vals))) # these are not used by SimpleRegressionPerfData
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, 
                             ids=ids, w=np.ones(len(ids)))
    # pass correct values through the transformers
    for t in perf.transformers:
        d = t.transform(d)

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
    script_path = os.path.dirname(os.path.realpath(__file__))
    res_dir = tempfile.mkdtemp()

    params = {"verbose": "True",
    "datastore": "False", 
    "save_results": "False", 
    "model_type": "NN", 
    "featurizer": "ecfp", 
    "prediction_type": "classification",
    "split_strategy": "train_valid_test", 
    "splitter": "random",
    "split_test_frac": "0.15", 
    "split_valid_frac": "0.15", 
    "transformers": "True", 
    "dataset_key": os.path.join(script_path,
        '../test_datasets/aurka_chembl_base_smiles_union.csv'),
    "id_col": "compound_id", 
    "response_cols":"active", 
    "smiles_col": "base_rdkit_smiles",
    "max_epochs":"2",
    "prediction_type": "classification",
    "result_dir":res_dir}

    # setup a pipeline that will be used to create performance data
    pparams = parse.wrapper(params)
    mp = model_pipeline.ModelPipeline(pparams)
    mp.train_model()

    # creat performance data
    perf = perf_data.create_perf_data(mp.params.prediction_type, 
            mp.data, mp.model_wrapper.transformers, 'train')

    ids = sorted(list(mp.data.train_valid_dsets[0][0].ids))
    weights = perf.get_weights()
    real_vals = perf.get_real_values()
    assert weights.shape == (len(ids),1)
    assert all(weights==1)

    d = dc.data.NumpyDataset(X=np.ones_like(real_vals), y=real_vals, ids=ids, w=np.ones(len(ids)))
    # There should be no transformers
    assert len(perf.transformers) == 0

    num_classes = 2
    # input to to_one_hot needs to have the shape (N,) not (N,1)
    pred_vals = dc.metrics.to_one_hot(d.y.reshape(len(d.y)), num_classes)
    # This should have r2 of 1
    roc_auc_score = perf.accumulate_preds(pred_vals)
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
    test_SimpleClassificationPerfData()
    #test_KFoldClassificationPerfData()
    #test_SimpleRegressionPerfData()
    #test_KFoldRegressionPerfData()