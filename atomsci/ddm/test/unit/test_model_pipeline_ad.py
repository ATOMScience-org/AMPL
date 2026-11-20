import numpy as np
from types import SimpleNamespace

from atomsci.ddm.pipeline.model_pipeline import (
    calc_AD_kmean_dist,
    calc_AD_kmean_local_density,
    drop_nan_training_rows_for_ad,
    get_invalid_pred_ad_penalty,
)
from atomsci.ddm.pipeline.model_datasets import should_persist_featurized_data


def test_drop_nan_training_rows_for_ad_filters_bad_rows_and_ids():
    train_dset = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0],
    ])
    ids = np.array(["cmpd_1", "cmpd_2", "cmpd_3"])

    filtered_dset, filtered_ids = drop_nan_training_rows_for_ad(train_dset, ids=ids)

    assert filtered_dset.shape == (2, 3)
    assert np.array_equal(filtered_dset, np.array([
        [1.0, 2.0, 3.0],
        [7.0, 8.0, 9.0],
    ]))
    assert np.array_equal(filtered_ids, np.array(["cmpd_1", "cmpd_3"]))


def test_drop_nan_training_rows_for_ad_preserves_clean_input():
    train_dset = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    filtered_dset, filtered_ids = drop_nan_training_rows_for_ad(train_dset)

    assert np.array_equal(filtered_dset, train_dset)
    assert filtered_ids is None


def test_should_persist_featurized_data_defaults_true_and_honors_override():
    assert should_persist_featurized_data(SimpleNamespace()) is True
    assert should_persist_featurized_data(
        SimpleNamespace(persist_featurized_data=False)
    ) is False


def test_calc_ad_kmean_dist_penalizes_prediction_rows_with_nan_features():
    train_dset = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    pred_dset = np.array([
        [0.5, 0.5],
        [np.nan, 0.5],
    ])

    ad_scores, nn_indexes = calc_AD_kmean_dist(train_dset, pred_dset, k=1)

    assert ad_scores.shape == (2,)
    assert nn_indexes.shape == (2, 1)
    assert ad_scores[0] < ad_scores[1]
    assert ad_scores[1] == get_invalid_pred_ad_penalty(np.array([ad_scores[0]]))
    assert nn_indexes[1][0] == -1


def test_calc_ad_kmean_local_density_penalizes_prediction_rows_with_nan_features():
    train_dset = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    pred_dset = np.array([
        [0.5, 0.5],
        [np.nan, 0.5],
    ])

    ad_scores = calc_AD_kmean_local_density(train_dset, pred_dset, k=1)

    assert ad_scores.shape == (2,)
    assert ad_scores[0] < ad_scores[1]
    assert ad_scores[1] == get_invalid_pred_ad_penalty(np.array([ad_scores[0]]))
