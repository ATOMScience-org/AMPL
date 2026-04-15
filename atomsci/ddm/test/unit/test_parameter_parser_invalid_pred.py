import pytest

import atomsci.ddm.pipeline.parameter_parser as parse


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
