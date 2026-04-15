"""Unit tests for build_optuna_suggest() and Optuna Study checkpoint round-trip."""

import math
import os
import pickle
import tempfile
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd

optuna = pytest.importorskip("optuna")

from atomsci.ddm.utils.hyperparam_search_wrapper import build_optuna_suggest
import atomsci.ddm.utils.hyperparam_search_wrapper as hsw


# ---------------------------------------------------------------------------
# build_optuna_suggest — per-method tests
# ---------------------------------------------------------------------------

def test_build_optuna_suggest_choice():
    trial = optuna.trial.FixedTrial({"lr": 0.001})
    result = build_optuna_suggest(trial, "lr", "choice", [0.0001, 0.001, 0.01])
    assert result == 0.001


def test_build_optuna_suggest_uniform():
    trial = optuna.trial.FixedTrial({"lr": 0.0005})
    result = build_optuna_suggest(trial, "lr", "uniform", [0.00001, 0.001])
    assert result == 0.0005


def test_build_optuna_suggest_loguniform():
    # param_list values are natural-log scale (legacy hyperopt convention);
    # build_optuna_suggest converts them via math.exp() before calling Optuna.
    log_low = math.log(0.001)
    log_high = math.log(1.0)
    expected = 0.01
    trial = optuna.trial.FixedTrial({"wdp": expected})
    result = build_optuna_suggest(trial, "wdp", "loguniform", [log_low, log_high])
    assert result == expected


def test_build_optuna_suggest_uniformint():
    trial = optuna.trial.FixedTrial({"rfe": 200})
    result = build_optuna_suggest(trial, "rfe", "uniformint", [64, 512])
    assert result == 200
    assert isinstance(result, int)


def test_build_optuna_suggest_invalid():
    trial = optuna.trial.FixedTrial({})
    with pytest.raises(ValueError):
        build_optuna_suggest(trial, "x", "badmethod", [0.0, 1.0])


# ---------------------------------------------------------------------------
# Optuna Study checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip():
    study = optuna.create_study(direction="minimize")

    def dummy_objective(trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        return x ** 2

    study.optimize(dummy_objective, n_trials=3)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        ckpt_path = f.name
    try:
        with open(ckpt_path, "wb") as f:
            pickle.dump(study, f)

        with open(ckpt_path, "rb") as f:
            loaded = pickle.load(f)

        assert len(loaded.trials) == len(study.trials)
        assert loaded.best_value == study.best_value
        assert loaded.direction == study.direction
    finally:
        os.remove(ckpt_path)


class _FakeTrial:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_int(self, name, low, high):
        return int(low)


class _FakeStudy:
    def __init__(self):
        self.trials = []

    def optimize(self, objective, n_trials):
        for _ in range(n_trials):
            trial = _FakeTrial()
            trial.value = objective(trial)
            self.trials.append(trial)


class _FakePerfData:
    def __init__(self, pred_type):
        self.pred_type = pred_type

    def get_prediction_results(self):
        if self.pred_type == 'regression':
            return {'r2_score': np.nan, 'rms_score': np.inf}
        return {'roc_auc_score': np.nan, 'accuracy_score': np.inf}


class _FakeModelWrapper:
    def __init__(self, pred_type):
        self.pred_type = pred_type

    def get_perf_data(self, subset=None, epoch_label=None):
        return _FakePerfData(self.pred_type)


class _FakeModelPipeline:
    def __init__(self, params):
        self.params = params
        self.model_wrapper = _FakeModelWrapper(params.prediction_type)

    def train_model(self):
        return None


def _build_optuna_params(tmpdir, prediction_type):
    return SimpleNamespace(
        result_dir=tmpdir,
        model_type='RF|1',
        prediction_type=prediction_type,
        featurizer='ecfp',
        descriptor_type='moe',
        hp_checkpoint_load=None,
        hp_checkpoint_save=None,
        rfe='choice|10,20',
        rfd='choice|3,4',
        rff='choice|5,6',
        rf_estimators=10,
        rf_max_depth=3,
        rf_max_features=5,
        lr=None,
        ls=None,
        ls_ratio=None,
        dp=None,
        wdp=None,
        wdt=None,
        xgbg=None,
        xgba=None,
        xgbb=None,
        xgbl=None,
        xgbd=None,
        xgbc=None,
        xgbs=None,
        xgbn=None,
        xgbw=None,
        layer_sizes='100,50',
        dropouts='0.1,0.1',
        learning_rate=0.001,
        weight_decay_penalty_type='l2',
        weight_decay_penalty=1e-05,
        hyperparam=True,
    )


def test_optuna_objective_sanitizes_non_finite_regression_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        params = _build_optuna_params(td, 'regression')

        monkeypatch.setattr(hsw.mp, 'ModelPipeline', _FakeModelPipeline)
        monkeypatch.setattr(hsw.parse, 'wrapper', lambda d: SimpleNamespace(**d, model_tarball_path='/tmp/fake_reg_model.tar.gz'))
        monkeypatch.setattr(hsw.optuna, 'create_study', lambda direction='minimize': _FakeStudy())

        search = hsw.OptunaSearch(params)
        search.run_search()

        perf_files = [f for f in os.listdir(td) if f.startswith('performance_regression_') and f.endswith('.csv')]
        assert len(perf_files) == 1
        perf_df = pd.read_csv(os.path.join(td, perf_files[0]))
        assert perf_df['valid_r2'].iloc[0] == 0
        assert perf_df['valid_rms'].iloc[0] == 100


def test_optuna_objective_sanitizes_non_finite_classification_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        params = _build_optuna_params(td, 'classification')

        monkeypatch.setattr(hsw.mp, 'ModelPipeline', _FakeModelPipeline)
        monkeypatch.setattr(hsw.parse, 'wrapper', lambda d: SimpleNamespace(**d, model_tarball_path='/tmp/fake_cls_model.tar.gz'))
        monkeypatch.setattr(hsw.optuna, 'create_study', lambda direction='minimize': _FakeStudy())

        search = hsw.OptunaSearch(params)
        search.run_search()

        perf_files = [f for f in os.listdir(td) if f.startswith('performance_classification_') and f.endswith('.csv')]
        assert len(perf_files) == 1
        perf_df = pd.read_csv(os.path.join(td, perf_files[0]))
        assert perf_df['valid_roc_auc'].iloc[0] == 0
        assert perf_df['valid_acc'].iloc[0] == 0
