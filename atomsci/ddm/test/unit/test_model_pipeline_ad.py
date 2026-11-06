import numpy as np
from types import SimpleNamespace

from atomsci.ddm.pipeline.model_pipeline import drop_nan_training_rows_for_ad
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
