import tempfile

import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import numpy as np
import os
import json

import logging
logger = logging.getLogger(__name__)

def test_balancing_transformer():
    dset_key = make_relative_to_file('../../test_datasets/MRP3_dataset.csv')

    res_dir = tempfile.mkdtemp()

    balanced_params = params_w_balan(dset_key, res_dir)
    balanced_weights = make_pipeline_and_get_weights(balanced_params)
    (major_weight, minor_weight), (major_count, minor_count) = np.unique(balanced_weights, return_counts=True)
    assert major_weight < minor_weight
    assert major_count > minor_count

    nonbalanced_params = params_wo_balan(dset_key, res_dir)
    nonbalanced_weights = make_pipeline_and_get_weights(nonbalanced_params)
    (weight,), (count,) = np.unique(nonbalanced_weights, return_counts=True)
    assert weight == 1
    assert count == 436

    smote_balanced_params = params_w_SMOTE_balan(dset_key, res_dir)
    smote_balanced_params['sampling_ratio'] = 0.5
    smote_balanced_weights = make_pipeline_and_get_weights(smote_balanced_params)
    (weight,), (count,) = np.unique(smote_balanced_weights, return_counts=True)
    # all weights should be the same
    assert np.all(weight==weight[0])

    smote_balanced_params = params_w_SMOTE_balan(dset_key, res_dir)
    smote_balanced_params['sampling_ratio'] = 0.8
    smote_balanced_weights = make_pipeline_and_get_weights(smote_balanced_params)
    (major_weight, minor_weight), (major_count, minor_count) = np.unique(smote_balanced_weights, return_counts=True)
    # There should be one weight that's larger and one that is smaller
    assert major_weight < minor_weight
    assert major_count > minor_count

def make_pipeline_and_get_weights(params):
    pparams = parse.wrapper(params)
    model_pipeline = mp.ModelPipeline(pparams)
    model_pipeline.train_model()

    return model_pipeline.data.train_valid_dsets[0][0].w

def make_relative_to_file(relative_path):
    script_path = os.path.dirname(os.path.realpath(__file__))
    result = os.path.join(script_path, relative_path)

    return result

def read_params(json_file, tmp_dskey, res_dir):
    with open(json_file, 'r') as file:
        params = json.load(file)
    params['result_dir'] = res_dir
    params['dataset_key'] = tmp_dskey
    return params

def params_wo_balan(dset_key, res_dir):
    # Train classification models without balancing weights. Repeat this several times so we can get some statistics on the performance metrics.
    params = read_params(
        make_relative_to_file('jsons/wo_balancing_transformer.json'),
        dset_key,
        res_dir)

    return params

def params_w_balan(dset_key, res_dir):
    # Now train models on the same dataset with balancing weights
    params = read_params(
        make_relative_to_file('jsons/balancing_transformer.json'),
        dset_key,
        res_dir
    )

    return params

def params_w_SMOTE_balan(dset_key, res_dir):
    # Try with SMOTE with ratio set to .50
    params = read_params(
        make_relative_to_file('jsons/SMOTE_balancing_transformer.json'),
        dset_key,
        res_dir
    )

    return params

if __name__ == '__main__':
    test_balancing_transformer()