import tempfile

import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.transformations as trans
import numpy as np
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import make_test_datasets


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

def test_all_transformers():
    res_dir = tempfile.mkdtemp()
    dskey = os.path.join(res_dir, 'special_test_dset.csv')
    params = params_w_balan(dskey, res_dir)
    make_test_datasets.make_test_dataset_and_split(dskey, params['descriptor_type'])

    params['previously_featurized'] = True
    params['previously_split'] = True
    params['splitter'] = 'index'
    params['split_uuid'] = 'testsplit'
    params['split_strategy'] = 'train_valid_test'
    params['response_cols'] = ['class']

    # check that the transformers are correct
    model_pipeline = make_pipeline(params)
    assert(len(model_pipeline.model_wrapper.transformers[0])==0)

    assert(len(model_pipeline.model_wrapper.transformers_x[0])==1)
    transformer_x = model_pipeline.model_wrapper.transformers_x[0][0]
    assert(isinstance(transformer_x, trans.NormalizationTransformerMissingData))

    assert(len(model_pipeline.model_wrapper.transformers_w[0])==1)
    assert(isinstance(model_pipeline.model_wrapper.transformers_w[0][0], trans.BalancingTransformer))

    # check that the transforms are correct
    train_dset = model_pipeline.data.train_valid_dsets[0][0]
    trans_train_dset = model_pipeline.model_wrapper.transform_dataset(train_dset, fold=0)

    # mean should be nearly 0
    assert abs(np.mean(trans_train_dset.X) - 0) < 1e-4
    np.testing.assert_array_almost_equal(transformer_x.X_means, np.zeros_like(transformer_x.X_means))
    # std should be nearly 2
    assert abs(np.std(trans_train_dset.X) - 1) < 1e-4
    np.testing.assert_array_almost_equal(transformer_x.X_stds, np.ones_like(transformer_x.X_stds)*2)
    # there is an 80/20 imbalance in classification
    weights = trans_train_dset.w
    (weight1, weight2), (count1, count2) = np.unique(weights, return_counts=True)
    assert (weight1*count1 - weight2*count2) < 1e-3

    # validation set has a different distribution
    valid_dset = model_pipeline.data.train_valid_dsets[0][1]
    trans_valid_dset = model_pipeline.model_wrapper.transform_dataset(valid_dset, fold=0)

    # untransformed mean is 10 expected transformed mean is (10 - 0) / 2
    assert abs(np.mean(trans_valid_dset.X) - 5) < 1e-4
    # untransformed std is 5 expected transformed std is 5/2 
    assert abs(np.std(trans_valid_dset.X) - (2.5))
    # validation has a 50/50 split. Majority class * 4 should equal oversampled minority class
    valid_weights = trans_valid_dset.w
    (valid_weight1, valid_weight2), (valid_count1, valid_count2) = np.unique(valid_weights, return_counts=True)
    assert (valid_weight1*valid_count1*4 - valid_weight2*valid_count2) < 1e-4


def make_pipeline(params):
    pparams = parse.wrapper(params)
    model_pipeline = mp.ModelPipeline(pparams)
    model_pipeline.train_model()
    
    return model_pipeline

def make_pipeline_and_get_weights(params):
    model_pipeline = make_pipeline(params)

    print(model_pipeline.model_wrapper.transformers_w)
    print(np.unique(model_pipeline.data.train_valid_dsets[0][0].y, return_counts=True))
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

if __name__ == '__main__':
    test_all_transformers()
    #test_balancing_transformer()