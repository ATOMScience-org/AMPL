import pytest
from types import SimpleNamespace

import atomsci.ddm.pipeline.parameter_parser as parse
from atomsci.ddm.pipeline import model_pipeline


def test_max_invalid_pred_frac_default_is_set():
    params = parse.wrapper({
        'dataset_key': '/tmp/dataset.csv',
        'bucket': 'public',
    })
    assert params.max_invalid_pred_frac == 0.01


def test_max_invalid_pred_frac_accepts_valid_values():
    params = parse.wrapper({
        'dataset_key': '/tmp/dataset.csv',
        'bucket': 'public',
        'max_invalid_pred_frac': 0.25,
    })
    assert params.max_invalid_pred_frac == 0.25


@pytest.mark.parametrize('bad_value', [-0.01, 1.01])
def test_max_invalid_pred_frac_rejects_out_of_range_values(bad_value):
    with pytest.raises(Exception, match='max_invalid_pred_frac must be between 0.0 and 1.0.'):
        parse.wrapper({
            'dataset_key': '/tmp/dataset.csv',
            'bucket': 'public',
            'max_invalid_pred_frac': bad_value,
        })


def test_create_model_metadata_persists_max_invalid_pred_frac():
    class _TestParams(SimpleNamespace):
        def __contains__(self, item):
            return hasattr(self, item)

    pipe = model_pipeline.ModelPipeline.__new__(model_pipeline.ModelPipeline)

    pipe.params = _TestParams(
        datastore=False,
        dataset_key='/tmp/dataset.csv',
        bucket='public',
        dataset_hash=None,
        id_col='compound_id',
        smiles_col='smiles',
        response_cols=['y'],
        robustscaler_with_centering='none',
        robustscaler_with_scaling='none',
        robustscaler_quartile_range='none',
        robustscaler_unit_variance='none',
        powertransformer_standardize='none',
        powertransformer_method='none',
        imputer_strategy='none',
        feature_transform_type='none',
        response_transform_type='none',
        weight_transform_type='none',
        result_dir='/tmp',
        production=False,
        model_bucket='public',
        system='local',
        model_type='RF',
        featurizer='ecfp',
        prediction_type='regression',
        model_choice_score_type='r2',
        max_invalid_pred_frac=0.25,
        num_model_tasks=1,
        class_number=2,
        transformers=[],
        transformer_key='none',
        transformer_bucket='public',
        transformer_oid='oid',
        uncertainty=False,
        save_results=False,
        hyperparam_uuid='hp-uuid',
        sampling_method=None,
        sampling_ratio=None,
        sampling_k_neighbors=None,
        model_uuid='model-uuid',
    )

    pipe.data = SimpleNamespace(
        dataset_oid='dataset-oid',
        get_split_metadata=lambda: {
            'splitter': 'random',
            'split_strategy': 'train_valid_test',
            'split_valid_frac': 0.1,
            'split_test_frac': 0.1,
            'split_uuid': 'split-uuid',
        },
        featurization=SimpleNamespace(get_feature_specific_metadata=lambda params: {}),
    )
    pipe.model_wrapper = SimpleNamespace(get_model_specific_metadata=lambda: {})
    pipe.seed = 7

    pipe.create_model_metadata()

    assert 'model_parameters' in pipe.model_metadata
    assert pipe.model_metadata['model_parameters']['max_invalid_pred_frac'] == 0.25
