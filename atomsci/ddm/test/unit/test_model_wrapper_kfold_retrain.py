"""
Unit tests for the k-fold "final retrain" branch of ForestModelWrapper.train() and
DCxgboostModelWrapper.train() (the RF and XGBoost wrappers share this logic).

When more than one fold is used, the wrappers retrain on the combined train+valid data and must
report train and test performance from that single retrained model (the one that gets saved), rather than
the accumulated per-fold predictions. These tests use stubbed wrapper internals so no model is
actually fit.
"""
import logging
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from atomsci.ddm.pipeline import model_wrapper as mw

WRAPPER_CLASSES = [mw.ForestModelWrapper, mw.DCxgboostModelWrapper]


class FakePerfData:
    """Minimal stand-in for PerfData that records accumulate_preds calls."""

    def __init__(self, subset, label):
        self.subset = subset
        self.label = label
        self.accumulated = []

    def accumulate_preds(self, preds, ids):
        self.accumulated.append((np.asarray(preds), np.asarray(ids)))
        return 0.5

    def compute_perf_metrics(self):
        return (f"{self.subset}-{self.label}-perf", f"{self.subset}-{self.label}-std")

    def model_choice_score(self, score_type):
        return 0.9


def make_dset(ids):
    return SimpleNamespace(ids=np.asarray(ids))


def build_wrapper(cls, num_folds):
    """Build a wrapper without running __init__, with every collaborator stubbed."""
    wrapper = cls.__new__(cls)
    wrapper.params = SimpleNamespace(prediction_type='regression', model_choice_score_type='r2')
    wrapper.log = logging.getLogger('test_kfold_retrain')
    wrapper.transformers = {k: f't{k}' for k in list(range(num_folds)) + ['final']}
    wrapper.model = mock.MagicMock()
    wrapper.model_save = mock.MagicMock()
    # Return a distinct dataset so tests detect predictions on untransformed data.
    wrapper.transform_dataset = mock.MagicMock(side_effect=lambda dset, fold: make_dset(dset.ids))

    final_test_ids = ['t0', 't1']
    wrapper.model.predict.side_effect = lambda dset, transformers: np.array([transformers] * len(dset.ids))

    train_valid = [(make_dset([f'tr{k}a', f'tr{k}b']), make_dset([f'va{k}'])) for k in range(num_folds)]
    data = SimpleNamespace(
        train_valid_dsets=train_valid,
        test_dset=make_dset(final_test_ids),
        combined_training_data=mock.MagicMock(return_value=make_dset(['all1', 'all2', 'all3'])),
    )
    pipeline = SimpleNamespace(data=data, metric_type='r2')
    return wrapper, pipeline


@pytest.fixture
def perf_stub():
    """Patch perf.create_perf_data and track every PerfData created (in call order)."""
    created = []

    def create(prediction_type, data, subset):
        pd_obj = FakePerfData(subset, label=f'obj{len(created)}')
        created.append(pd_obj)
        return pd_obj

    with mock.patch.object(mw.perf, 'create_perf_data', side_effect=create):
        yield created


@pytest.mark.parametrize('cls', WRAPPER_CLASSES)
def test_kfold_final_train_and_test_perf_come_from_retrained_model(cls, perf_stub):
    wrapper, pipeline = build_wrapper(cls, num_folds=3)
    wrapper.train(pipeline)

    # Initial fold aggregates plus fresh train/test results for the final model.
    assert [p.subset for p in perf_stub] == ['train', 'valid', 'test', 'train', 'test']
    fold_train_perf, fold_test_perf = perf_stub[0], perf_stub[2]
    final_train_perf, final_test_perf = perf_stub[3], perf_stub[4]

    assert len(fold_train_perf.accumulated) == 3
    assert wrapper.train_perf_data is final_train_perf
    assert len(final_train_perf.accumulated) == 1
    train_preds, train_ids = final_train_perf.accumulated[0]
    assert list(train_preds) == ['tfinal'] * 3
    assert list(train_ids) == ['all1', 'all2', 'all3']

    # the per-fold test PerfData accumulated one prediction set per fold ...
    assert len(fold_test_perf.accumulated) == 3
    # ... while the replacement saw exactly one pass, made with the 'final' transformers
    assert wrapper.test_perf_data is final_test_perf
    assert len(final_test_perf.accumulated) == 1
    preds, ids = final_test_perf.accumulated[0]
    assert list(preds) == ['tfinal', 'tfinal']
    assert list(ids) == ['t0', 't1']

    # reported test metrics are those of the retrained model, not the fold average
    assert wrapper.test_perf == 'test-obj4-perf'
    assert wrapper.test_perf_std == 'test-obj4-std'
    assert wrapper.train_perf == 'train-obj3-perf'
    assert wrapper.train_perf_std == 'train-obj3-std'
    # Validation and model selection remain the cross-validation aggregates.
    assert wrapper.valid_perf_data is perf_stub[1]
    assert len(wrapper.valid_perf_data.accumulated) == 3
    assert wrapper.valid_perf == 'valid-obj1-perf'
    assert wrapper.valid_perf_std == 'valid-obj1-std'
    assert wrapper.model_choice_score == 0.9


@pytest.mark.parametrize('cls', WRAPPER_CLASSES)
def test_kfold_retrains_on_combined_data_with_final_fold(cls, perf_stub):
    wrapper, pipeline = build_wrapper(cls, num_folds=2)
    calls = mock.MagicMock()
    calls.attach_mock(wrapper.model.fit, 'fit')
    calls.attach_mock(wrapper.model.predict, 'predict')
    calls.attach_mock(wrapper.model_save, 'save')

    def check_saved_metrics():
        assert wrapper.train_perf_data is perf_stub[3]
        assert wrapper.test_perf_data is perf_stub[4]
        assert wrapper.train_perf == 'train-obj3-perf'
        assert wrapper.test_perf == 'test-obj4-perf'

    wrapper.model_save.side_effect = check_saved_metrics
    wrapper.train(pipeline)

    pipeline.data.combined_training_data.assert_called_once_with()
    # one fit per fold plus the final fit on combined data
    assert wrapper.model.fit.call_count == 3
    final_fit_dset = wrapper.model.fit.call_args_list[-1].args[0]
    assert list(final_fit_dset.ids) == ['all1', 'all2', 'all3']
    assert final_fit_dset is not pipeline.data.combined_training_data.return_value
    final_predictions = wrapper.model.predict.call_args_list[-2:]
    assert final_predictions[0].args[0] is final_fit_dset
    assert final_predictions[0].args[1] == 'tfinal'
    assert final_predictions[1].args[0] is not pipeline.data.test_dset
    assert final_predictions[1].args[1] == 'tfinal'
    assert [call[0] for call in calls.mock_calls[-4:]] == ['fit', 'predict', 'predict', 'save']
    # the retrained model's predictions use the 'final' transformers
    final_folds = [c.kwargs.get('fold') for c in wrapper.transform_dataset.call_args_list]
    assert final_folds.count('final') == 2  # combined training data + test set
    wrapper.model_save.assert_called_once_with()
    assert wrapper.best_epoch == 0
    assert wrapper.model_choice_score == 0.9


@pytest.mark.parametrize('cls', WRAPPER_CLASSES)
def test_single_fold_does_not_retrain_or_replace_train_and_test_perf(cls, perf_stub):
    wrapper, pipeline = build_wrapper(cls, num_folds=1)
    wrapper.train(pipeline)

    # Only initial train/valid/test PerfData; no replacements.
    assert [p.subset for p in perf_stub] == ['train', 'valid', 'test']
    assert wrapper.train_perf_data is perf_stub[0]
    assert wrapper.valid_perf_data is perf_stub[1]
    assert wrapper.test_perf_data is perf_stub[2]
    for subset, label in [('train', 'obj0'), ('valid', 'obj1'), ('test', 'obj2')]:
        assert getattr(wrapper, f'{subset}_perf') == f'{subset}-{label}-perf'
        assert getattr(wrapper, f'{subset}_perf_std') == f'{subset}-{label}-std'
        assert len(getattr(wrapper, f'{subset}_perf_data').accumulated) == 1
    assert wrapper.model.predict.call_count == 3
    assert all(c.kwargs.get('fold') != 'final' for c in wrapper.transform_dataset.call_args_list)
    pipeline.data.combined_training_data.assert_not_called()
    assert wrapper.model.fit.call_count == 1
    wrapper.model_save.assert_called_once_with()
    assert wrapper.best_epoch == 0
