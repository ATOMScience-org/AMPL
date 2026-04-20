"""Unit tests for build_optuna_suggest() and Optuna Study checkpoint round-trip."""

import math
import os
import pickle
import tempfile

import optuna
import pytest

from atomsci.ddm.utils.hyperparam_search_wrapper import build_optuna_suggest


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
