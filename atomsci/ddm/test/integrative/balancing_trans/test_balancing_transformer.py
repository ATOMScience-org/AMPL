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
    """
    Test the balancing transformer to ensure that it correctly adjusts weights for imbalanced datasets.
    """
    dset_key = make_relative_to_file('../../test_datasets/MRP3_dataset.csv')

    res_dir = tempfile.mkdtemp()

    print('-=======normal balancing===================================')
    balanced_params = params_w_balan(dset_key, res_dir)
    balanced_weights = make_pipeline_and_get_weights(balanced_params)
    (major_weight, minor_weight), (major_count, minor_count) = np.unique(balanced_weights, return_counts=True)
    assert major_weight < minor_weight
    assert major_count > minor_count
    print('-==========================================')

    print('-=======no balancing===================================')
    nonbalanced_params = params_wo_balan(dset_key, res_dir)
    nonbalanced_weights = make_pipeline_and_get_weights(nonbalanced_params)
    (weight,), (count,) = np.unique(nonbalanced_weights, return_counts=True)
    assert weight == 1
    print('-==========================================')

    print('-=======SMOTE balancing===================================')
    smote_balanced_params = params_w_SMOTE_balan(dset_key, res_dir)
    smote_balanced_params['sampling_ratio'] = 1
    print('sampling_ratio: ', smote_balanced_params['sampling_ratio'])
    smote_balanced_weights = make_pipeline_and_get_weights(smote_balanced_params)
    # all weights should be the same
    (weight1,), (count1,)= np.unique(smote_balanced_weights, return_counts=True)
    print('-==========================================')

    print('-=======SMOTE 0.5 balancing===================================')
    smote_balanced_params = params_w_SMOTE_balan(dset_key, res_dir)
    smote_balanced_params['sampling_ratio'] = 0.5
    smote_balanced_weights = make_pipeline_and_get_weights(smote_balanced_params)
    (major_weight, minor_weight), (major_count, minor_count) = np.unique(smote_balanced_weights, return_counts=True)
    # there should be twice as many major class as minor class
    assert abs((major_weight*2) - minor_weight) < .0001
    assert abs(major_count - (minor_count * 2)) < .0001
    print('-==========================================')

def params_w_SMOTE_balan(dset_key, res_dir):
    # Try with SMOTE with ratio set to .50
    params = read_params(
        make_relative_to_file('jsons/SMOTE_balancing_transformer.json'),
        dset_key,
        res_dir
    )

    return params

def test_all_transformers():
    """
    Test all transformers to ensure they work correctly with the dataset.

    """
    res_dir = tempfile.mkdtemp()
    dskey = os.path.join(res_dir, 'special_test_dset.csv')
    params = read_params(
        make_relative_to_file('jsons/all_transforms.json'),
        dskey,
        res_dir
    )
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

    # transformed validation mean is 10 expected transformed mean is (10 - 0) / 2
    assert abs(np.mean(trans_valid_dset.X) - 5) < 1e-4
    # transformed validation std is 5 expected transformed std is 5/2 
    assert abs(np.std(trans_valid_dset.X) - (2.5)) < 1e-4
    # validation has a 50/50 split. Majority class * 4 should equal oversampled minority class
    valid_weights = trans_valid_dset.w
    (valid_weight1, valid_weight2), (valid_count1, valid_count2) = np.unique(valid_weights, return_counts=True)
    assert (valid_weight1*valid_count1*4 - valid_weight2*valid_count2) < 1e-4

def make_pipeline(params):
    """
    Generates a pipeline given parameters
    """
    pparams = parse.wrapper(params)
    model_pipeline = mp.ModelPipeline(pparams)
    model_pipeline.train_model()
    
    return model_pipeline

def make_pipeline_and_get_weights(params):
    """
    Generates the pipeline and gets the weights given parameters
    """
    model_pipeline = make_pipeline(params)
    model_wrapper = model_pipeline.model_wrapper
    train_dataset = model_pipeline.data.train_valid_dsets[0][0]
    transformed_data = model_wrapper.transform_dataset(train_dataset, fold=0)

    return transformed_data.w

def make_relative_to_file(relative_path):
    """
    Generates the full path relative to the location of this file.
    """
    
    script_path = os.path.dirname(os.path.realpath(__file__))
    result = os.path.join(script_path, relative_path)

    return result

def read_params(json_file, tmp_dskey, res_dir):
    """
    Read parameters from a JSON file and update them with the dataset key and result directory.

    Parameters:
    json_file (str): Path to the JSON file containing parameters.
    tmp_dskey (str): Temporary dataset key.
    res_dir (str): Result directory.

    Returns:
    dict: Updated parameters.
    """
    with open(json_file, 'r') as file:
        params = json.load(file)
    params['result_dir'] = res_dir
    params['dataset_key'] = tmp_dskey
    return params

def params_wo_balan(dset_key, res_dir):
    """
    Reads params for models without balancing weight
    """
    params = read_params(
        make_relative_to_file('jsons/wo_balancing_transformer.json'),
        dset_key,
        res_dir)

    return params

def params_w_balan(dset_key, res_dir):
    """
    Reads params for models with balancing weight
    """
    params = read_params(
        make_relative_to_file('jsons/balancing_transformer.json'),
        dset_key,
        res_dir
    )

    return params

def test_kfold_transformers():
    """
    Test transformers for a kfold classification model
    """
    res_dir = tempfile.mkdtemp()
    dskey = os.path.join(res_dir, 'special_test_dset.csv')
    params = read_params(
        make_relative_to_file('jsons/all_transforms.json'),
        dskey,
        res_dir
    )
    num_folds = 3

    make_test_datasets.make_kfold_dataset_and_split(dskey, 
            params['descriptor_type'], num_folds=num_folds)

    params['previously_featurized'] = True
    params['previously_split'] = True
    params['splitter'] = 'index'
    params['split_uuid'] = 'testsplit'
    params['split_strategy'] = 'k_fold_cv'
    params['num_folds'] = num_folds
    params['response_cols'] = ['class']

    # check that the transformers are correct
    model_pipeline = make_pipeline(params)
    for f in range(num_folds):
        assert(len(model_pipeline.model_wrapper.transformers[f])==0)

        assert(len(model_pipeline.model_wrapper.transformers_x[f])==1)
        transformer_x = model_pipeline.model_wrapper.transformers_x[f][0]
        assert(isinstance(transformer_x, trans.NormalizationTransformerMissingData))

        assert(len(model_pipeline.model_wrapper.transformers_w[f])==1)
        assert(isinstance(model_pipeline.model_wrapper.transformers_w[f][0], trans.BalancingTransformer))

    assert(len(model_pipeline.data.train_valid_dsets)==num_folds)
    for i, (train_dset, valid_dset) in enumerate(model_pipeline.data.train_valid_dsets):
        # check that the transforms are correct
        trans_train_dset = model_pipeline.model_wrapper.transform_dataset(train_dset, fold=i)
        transformer_x = model_pipeline.model_wrapper.transformers_x[i][0]

        # the mean of each fold is the square of the fold value
        fold_means = [f*f for f in range(num_folds)]
        fold_means.remove(i*i)
        expected_mean = sum(fold_means)/len(fold_means)

        # mean should be nearly 0
        assert abs(np.mean(trans_train_dset.X)) < 1e-4

        # transformer means should be around expected_mean
        np.testing.assert_array_almost_equal(transformer_x.X_means, np.ones_like(transformer_x.X_means)*expected_mean)

        # std should be nearly 1
        assert abs(np.std(trans_train_dset.X) - 1) < 1e-4

        # there is an 80/20 imbalance in classification
        weights = trans_train_dset.w
        (weight1, weight2), (count1, count2) = np.unique(weights, return_counts=True)
        assert (weight1*count1 - weight2*count2) < 1e-3

        # validation set has a different distribution
        trans_valid_dset = model_pipeline.model_wrapper.transform_dataset(valid_dset, fold=i)

        # validation mean should not be 0
        assert abs(np.mean(trans_valid_dset.X)) > 1e-4

        # validation has a 80/20 split. Majority class * 4 should equal oversampled minority class
        valid_weights = trans_valid_dset.w
        (valid_weight1, valid_weight2), (valid_count1, valid_count2) = np.unique(valid_weights, return_counts=True)
        assert (valid_weight1*valid_count1 - valid_weight2*valid_count2) < 1e-3

    # test that the final transformer is correct
    expected_mean = sum([f*f for f in range(num_folds)])/num_folds
    trans_combined_dset = model_pipeline.model_wrapper.transform_dataset(
        model_pipeline.data.combined_training_data(), fold='final')
    # mean should be nearly 0
    assert abs(np.mean(trans_combined_dset.X)) < 1e-4

    assert(len(model_pipeline.model_wrapper.transformers_x['final'])==1)
    transformer_x = model_pipeline.model_wrapper.transformers_x['final'][0]
    assert(isinstance(transformer_x, trans.NormalizationTransformerMissingData))
    # transformer means should be around expected_mean
    np.testing.assert_array_almost_equal(transformer_x.X_means, np.ones_like(transformer_x.X_means)*expected_mean)

def test_kfold_regression_transformers():
    """
    Tests transformers for each fold of a kfold regression model. Ensures
    that the transformers are correct for each fold.
    """
    res_dir = tempfile.mkdtemp()
    dskey = os.path.join(res_dir, 'special_test_dset.csv')
    params = read_params(
        make_relative_to_file('jsons/all_transforms_regression.json'),
        dskey,
        res_dir
    )
    num_folds = 3

    make_test_datasets.make_kfold_dataset_and_split(dskey, 
            params['descriptor_type'], num_folds=num_folds)

    # check that the transformers are correct
    model_pipeline = make_pipeline(params)
    for f in range(num_folds):
        assert(len(model_pipeline.model_wrapper.transformers[f])==1)
        transformer = model_pipeline.model_wrapper.transformers[f][0]
        assert(isinstance(transformer, trans.NormalizationTransformerMissingData))


        assert(len(model_pipeline.model_wrapper.transformers_x[f])==1)
        transformer_x = model_pipeline.model_wrapper.transformers_x[f][0]
        assert(isinstance(transformer_x, trans.NormalizationTransformerMissingData))

        assert(len(model_pipeline.model_wrapper.transformers_w[f])==0)

    assert(len(model_pipeline.data.train_valid_dsets)==num_folds)
    for i, (train_dset, valid_dset) in enumerate(model_pipeline.data.train_valid_dsets):
        # check that the transforms are correct
        trans_train_dset = model_pipeline.model_wrapper.transform_dataset(train_dset, fold=i)
        transformer = model_pipeline.model_wrapper.transformers[i][0]

        # the mean of each fold is the square of the fold value
        fold_means = [f*f for f in range(num_folds)]
        fold_means.remove(i*i)
        expected_mean_1 = sum(fold_means)/len(fold_means)
        expected_mean_2 = expected_mean_1*10

        # mean should be nearly 0
        assert abs(np.mean(trans_train_dset.y)) < 1e-4

        # transformer means should be around expected_mean
        assert(transformer.y_means.shape==(2,))
        expected_y_means = np.array([expected_mean_1, expected_mean_2])
        np.testing.assert_array_almost_equal(transformer.y_means, expected_y_means)

        # std should be nearly 1
        assert abs(np.std(trans_train_dset.y) - 1) < 1e-4

        # validation set has a different distribution
        trans_valid_dset = model_pipeline.model_wrapper.transform_dataset(valid_dset, fold=i)

        # validation mean should not be 0
        assert abs(np.mean(trans_valid_dset.y)) > 1e-4

    # test that the final transformer is correct
    expected_mean_1 = sum([f*f for f in range(num_folds)])/num_folds
    expected_mean_2 = expected_mean_1*10
    expected_y_means = np.array([expected_mean_1, expected_mean_2])

    trans_combined_dset = model_pipeline.model_wrapper.transform_dataset(
        model_pipeline.data.combined_training_data(), fold='final')

    # mean should be nearly 0
    assert abs(np.mean(trans_combined_dset.y)) < 1e-4

    assert(len(model_pipeline.model_wrapper.transformers['final'])==1)
    transformer = model_pipeline.model_wrapper.transformers['final'][0]
    assert(isinstance(transformer, trans.NormalizationTransformerMissingData))
    # transformer means should be around expected_mean
    np.testing.assert_array_almost_equal(transformer.y_means, expected_y_means)


if __name__ == '__main__':
    test_kfold_regression_transformers()
    test_kfold_transformers()
    test_all_transformers()
    test_balancing_transformer()