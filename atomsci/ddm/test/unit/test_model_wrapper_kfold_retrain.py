"""
Unit tests for the k-fold "final retrain" branch of ForestModelWrapper.train() and
DCxgboostModelWrapper.train() (the RF and XGBoost wrappers share this logic).

When more than one fold is used, the wrappers retrain on the combined train+valid data and must
report test performance from that single retrained model (the one that gets saved), rather than
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
    # transform_dataset is the identity; record the fold labels it was called with
    wrapper.transform_dataset = mock.MagicMock(side_effect=lambda dset, fold: dset)

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
def test_kfold_final_test_perf_comes_from_retrained_model(cls, perf_stub):
    wrapper, pipeline = build_wrapper(cls, num_folds=3)
    wrapper.train(pipeline)

    # 3 PerfData up front (train/valid/test) + 1 fresh test PerfData for the retrained model
    assert [p.subset for p in perf_stub] == ['train', 'valid', 'test', 'test']
    fold_test_perf, final_test_perf = perf_stub[2], perf_stub[3]

    # the per-fold test PerfData accumulated one prediction set per fold ...
    assert len(fold_test_perf.accumulated) == 3
    # ... while the replacement saw exactly one pass, made with the 'final' transformers
    assert wrapper.test_perf_data is final_test_perf
    assert len(final_test_perf.accumulated) == 1
    preds, ids = final_test_perf.accumulated[0]
    assert list(preds) == ['tfinal', 'tfinal']
    assert list(ids) == ['t0', 't1']

    # reported test metrics are those of the retrained model, not the fold average
    assert wrapper.test_perf == 'test-obj3-perf'
    assert wrapper.test_perf_std == 'test-obj3-std'
    # train/valid metrics remain the cross-validation aggregates
    assert wrapper.train_perf == 'train-obj0-perf'
    assert wrapper.valid_perf == 'valid-obj1-perf'


@pytest.mark.parametrize('cls', WRAPPER_CLASSES)
def test_kfold_retrains_on_combined_data_with_final_fold(cls, perf_stub):
    wrapper, pipeline = build_wrapper(cls, num_folds=2)
    wrapper.train(pipeline)

    pipeline.data.combined_training_data.assert_called_once_with()
    # one fit per fold plus the final fit on combined data
    assert wrapper.model.fit.call_count == 3
    final_fit_dset = wrapper.model.fit.call_args_list[-1].args[0]
    assert list(final_fit_dset.ids) == ['all1', 'all2', 'all3']
    # the retrained model's predictions use the 'final' transformers
    final_folds = [c.kwargs.get('fold') for c in wrapper.transform_dataset.call_args_list]
    assert final_folds.count('final') == 2  # combined training data + test set
    wrapper.model_save.assert_called_once_with()
    assert wrapper.best_epoch == 0
    assert wrapper.model_choice_score == 0.9


@pytest.mark.parametrize('cls', WRAPPER_CLASSES)
def test_single_fold_does_not_retrain_or_replace_test_perf(cls, perf_stub):
    wrapper, pipeline = build_wrapper(cls, num_folds=1)
    wrapper.train(pipeline)

    # only the initial train/valid/test PerfData; no replacement test PerfData
    assert [p.subset for p in perf_stub] == ['train', 'valid', 'test']
    assert wrapper.test_perf_data is perf_stub[2]
    assert wrapper.test_perf == 'test-obj2-perf'
    pipeline.data.combined_training_data.assert_not_called()
    assert wrapper.model.fit.call_count == 1
    wrapper.model_save.assert_called_once_with()
    assert wrapper.best_epoch == 0
