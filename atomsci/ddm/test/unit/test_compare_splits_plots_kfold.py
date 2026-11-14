"""
Unit tests for the k-fold support in atomsci.ddm.utils.compare_splits_plots.SplitStats, plus the
split-mode inference, guard methods, umap/fraction plots and the standard-split branches of
make_all_plots that were previously uncovered.
"""
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from atomsci.ddm.utils import compare_splits_plots as csp
from atomsci.ddm.utils.compare_splits_plots import SplitStats, save_figure

SMILES = [
    'CCO', 'CCN', 'CCC', 'CCCl', 'CCBr', 'CCCC', 'CCCO', 'CCCN',
    'c1ccccc1', 'c1ccccc1C', 'c1ccccc1O', 'c1ccccc1N', 'CC(C)C', 'CC(C)O', 'CC(=O)O',
]
N_TEST = 3
N_FOLDS = 3


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def total_df():
    n = len(SMILES)
    return pd.DataFrame({
        'cmpd_id': [f'c{i}' for i in range(n)],
        'smiles': SMILES,
        'response': np.arange(n) % 2,
    })


@pytest.fixture
def kfold_split_df(total_df):
    """First 12 compounds are train_valid in 3 folds (0,1,2 repeating); the last 3 are test."""
    n_tv = len(total_df) - N_TEST
    subsets = ['train_valid'] * n_tv + ['test'] * N_TEST
    folds = [i % N_FOLDS for i in range(n_tv)] + [np.nan] * N_TEST
    return pd.DataFrame({'cmpd_id': total_df['cmpd_id'], 'subset': subsets, 'fold': folds})


@pytest.fixture
def standard_split_df(total_df):
    subsets = ['train'] * 9 + ['valid'] * 3 + ['test'] * 3
    return pd.DataFrame({'cmpd_id': total_df['cmpd_id'], 'subset': subsets})


@pytest.fixture
def kfold_ss(total_df, kfold_split_df):
    return SplitStats(total_df, kfold_split_df, smiles_col='smiles', id_col='cmpd_id',
                      response_cols=['response'])


@pytest.fixture
def standard_ss(total_df, standard_split_df):
    return SplitStats(total_df, standard_split_df, smiles_col='smiles', id_col='cmpd_id',
                      response_cols=['response'])


# --- split mode inference ---------------------------------------------------------------------
def test_infer_split_mode_standard(standard_split_df):
    assert SplitStats._infer_split_mode(standard_split_df) == 'standard'


def test_infer_split_mode_kfold(kfold_split_df):
    assert SplitStats._infer_split_mode(kfold_split_df) == 'kfold'


def test_infer_split_mode_kfold_requires_fold_column(kfold_split_df):
    with pytest.raises(ValueError, match="Cannot infer split mode"):
        SplitStats._infer_split_mode(kfold_split_df.drop(columns='fold'))


def test_infer_split_mode_unknown_subsets():
    df = pd.DataFrame({'cmpd_id': ['a', 'b'], 'subset': ['train', 'test']})
    with pytest.raises(ValueError, match="Cannot infer split mode"):
        SplitStats._infer_split_mode(df)


def test_init_unknown_split_mode_raises(total_df, standard_split_df, monkeypatch):
    monkeypatch.setattr(SplitStats, '_infer_split_mode', staticmethod(lambda df: 'bogus'))
    with pytest.raises(ValueError, match="Unknown split mode: bogus"):
        SplitStats(total_df, standard_split_df, 'smiles', 'cmpd_id', ['response'])


# --- initialization ---------------------------------------------------------------------------
def test_standard_init_sets_standard_attributes(standard_ss):
    assert standard_ss.split_mode == 'standard'
    assert not standard_ss.is_kfold
    assert standard_ss.n_folds is None
    assert standard_ss.fold_dists is None
    assert len(standard_ss.train_df) == 9
    assert len(standard_ss.valid_df) == 3
    assert len(standard_ss.test_df) == 3


def test_kfold_init_attributes(kfold_ss):
    assert kfold_ss.split_mode == 'kfold'
    assert kfold_ss.is_kfold
    assert kfold_ss.n_folds == N_FOLDS
    assert len(kfold_ss.test_df) == N_TEST
    assert set(kfold_ss.test_df['cmpd_id']) == {'c12', 'c13', 'c14'}
    # standard-split attributes are cleared for k-fold
    for attr in ('train_df', 'valid_df', 'dists_tvt', 'dists_tvv', 'train_fracs',
                 'valid_fracs', 'test_fracs', 'train_y', 'valid_y'):
        assert getattr(kfold_ss, attr) is None


def test_kfold_fold_dists_cover_every_fold(kfold_ss):
    assert list(kfold_ss.fold_dists.columns) == ['fold', 'distance']
    assert sorted(kfold_ss.fold_dists['fold'].unique()) == [0, 1, 2]
    # one nearest-neighbour distance per test compound per fold
    assert (kfold_ss.fold_dists.groupby('fold').size() == N_TEST).all()
    assert kfold_ss.fold_dists['distance'].between(0, 1).all()


def test_kfold_trainvalid_dists_have_one_row_per_validation_compound(kfold_ss):
    tvv = kfold_ss.fold_trainvalid_dists
    # 12 train_valid compounds, split 4/4/4 over 3 folds
    assert (tvv.groupby('fold').size() == 4).all()
    assert tvv['distance'].between(0, 1).all()


def test_kfold_training_set_is_complement_of_validation_fold(total_df, kfold_split_df, monkeypatch):
    """Regression test for the Copilot/PR review comment: the training set for fold f must be
    every train_valid compound NOT in fold f, and the validation set must be fold f only."""
    seen = []
    orig = SplitStats._get_dists

    def spy(self, df_a, df_b):
        seen.append((set(df_a['cmpd_id']), set(df_b['cmpd_id'])))
        return orig(self, df_a, df_b)

    monkeypatch.setattr(SplitStats, '_get_dists', spy)
    SplitStats(total_df, kfold_split_df, 'smiles', 'cmpd_id', ['response'])

    tv = kfold_split_df[kfold_split_df['subset'] == 'train_valid']
    test_ids = set(kfold_split_df.loc[kfold_split_df['subset'] == 'test', 'cmpd_id'])
    # _get_dists is called as (test, train) then (valid, train) for each fold
    assert len(seen) == 2 * N_FOLDS
    for i, fold in enumerate(sorted(tv['fold'].unique())):
        expected_valid = set(tv.loc[tv['fold'] == fold, 'cmpd_id'])
        expected_train = set(tv.loc[tv['fold'] != fold, 'cmpd_id'])
        (test_a, train_a), (valid_b, train_b) = seen[2 * i], seen[2 * i + 1]
        assert test_a == test_ids
        assert train_a == expected_train
        assert valid_b == expected_valid
        assert train_b == expected_train
        assert not (train_b & valid_b)


def test_kfold_fold_fracs_structure(kfold_ss):
    fracs = kfold_ss.fold_fracs
    assert set(fracs['subset']) == {'train', 'valid', 'test'}
    assert set(fracs.loc[fracs['subset'] == 'test', 'fold']) == {'all'}
    assert set(fracs.loc[fracs['subset'] == 'train', 'fold']) == {0, 1, 2}
    assert set(fracs.loc[fracs['subset'] == 'valid', 'fold']) == {0, 1, 2}
    assert fracs['frac'].between(0, 1).all()


# --- guards -----------------------------------------------------------------------------------
@pytest.mark.parametrize('method, args', [
    ('print_stats', ()),
    ('dist_hist_train_v_test_plot', ()),
    ('dist_hist_train_v_valid_plot', ()),
    ('dist_hist_plot', ([0.1, 0.2], 'title')),
])
def test_standard_only_methods_raise_for_kfold(kfold_ss, method, args):
    with pytest.raises(ValueError, match="not available for k-fold"):
        getattr(kfold_ss, method)(*args)


@pytest.mark.parametrize('method', ['dist_hist_kfold_plot', 'dist_hist_kfold_tvv_plot'])
def test_kfold_only_methods_raise_for_standard(standard_ss, method):
    with pytest.raises(ValueError, match="only available for k-fold"):
        getattr(standard_ss, method)()


@pytest.mark.parametrize('method', ['dist_hist_kfold_plot', 'dist_hist_kfold_tvv_plot'])
@pytest.mark.parametrize('bad_bins', [5, 15, 0])
def test_kfold_hist_rejects_invalid_bin_count(kfold_ss, method, bad_bins):
    with pytest.raises(ValueError, match="n_bins must be 10 or 20"):
        getattr(kfold_ss, method)(n_bins=bad_bins)


# --- k-fold plots -----------------------------------------------------------------------------
@pytest.mark.parametrize('method, title', [
    ('dist_hist_kfold_plot', 'k-fold Train vs Test Tanimoto Distance'),
    ('dist_hist_kfold_tvv_plot', 'k-fold Train vs Valid Tanimoto Distance'),
])
@pytest.mark.parametrize('n_bins', [10, 20])
def test_kfold_histogram_plots(kfold_ss, method, title, n_bins):
    _fig, ax = plt.subplots()
    out = getattr(kfold_ss, method)(n_bins=n_bins, ax=ax)

    assert out is ax
    assert ax.get_title() == title
    assert ax.get_xlabel() == 'Tanimoto distance'
    assert ax.get_ylabel() == 'Proportion of compounds'
    assert ax.get_xlim() == (0, 1)
    assert ax.get_legend().get_title().get_text() == 'Fold'
    # one legend entry per fold
    assert len(ax.get_legend().get_texts()) == N_FOLDS


def test_kfold_histogram_creates_axes_when_none_given(kfold_ss):
    plt.figure()
    ax = kfold_ss.dist_hist_kfold_plot()
    assert ax is not None


# --- umap / fraction plots --------------------------------------------------------------------
def test_umap_plot_kfold_uses_fold_hue_and_saves(kfold_ss, tmp_path):
    prefix = str(tmp_path / 'kf')
    kfold_ss.umap_plot(dist_path=prefix)
    assert os.path.exists(prefix + '_umap_scatter.png')
    assert os.path.exists(prefix + '_umap_scatter.svg')


def test_umap_plot_standard_uses_subset_hue_and_saves(standard_ss, tmp_path):
    prefix = str(tmp_path / 'std')
    standard_ss.umap_plot(dist_path=prefix)
    assert os.path.exists(prefix + '_umap_scatter.png')


def test_umap_plot_without_path_leaves_figure_open(kfold_ss):
    plt.close('all')
    kfold_ss.umap_plot()
    # the figure is intentionally left open for the caller when no path is given
    assert len(plt.get_fignums()) == 1


def test_subset_frac_plot_kfold(kfold_ss, tmp_path):
    prefix = str(tmp_path / 'kf')
    kfold_ss.subset_frac_plot(dist_path=prefix)
    assert os.path.exists(prefix + '_frac_box.png')


def test_subset_frac_plot_standard_list_response(standard_ss, tmp_path):
    prefix = str(tmp_path / 'std')
    standard_ss.subset_frac_plot(dist_path=prefix)
    assert os.path.exists(prefix + '_frac_box.png')


# --- make_all_plots ---------------------------------------------------------------------------
def test_make_all_plots_kfold_saves_all_figures(kfold_ss, tmp_path):
    prefix = str(tmp_path / 'kf')
    kfold_ss.make_all_plots(dist_path=prefix)
    for suffix in ('_kfold_tvt', '_kfold_tvv', '_umap_scatter', '_frac_box'):
        assert os.path.exists(prefix + suffix + '.png'), suffix
        assert os.path.exists(prefix + suffix + '.svg'), suffix


def test_make_all_plots_kfold_without_path_saves_nothing(kfold_ss, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kfold_ss.make_all_plots()
    assert list(tmp_path.iterdir()) == []


def test_make_all_plots_standard_saves_all_figures(standard_ss, tmp_path):
    prefix = str(tmp_path / 'std')
    standard_ss.make_all_plots(dist_path=prefix)
    for suffix in ('_tvt_dist_hist', '_tvv_dist_hist', '_umap_scatter', '_frac_box'):
        assert os.path.exists(prefix + suffix + '.png'), suffix


def test_make_all_plots_routes_by_split_mode(kfold_ss, standard_ss, monkeypatch):
    calls = []
    monkeypatch.setattr(SplitStats, 'umap_plot', lambda self, p='': calls.append('umap'))
    monkeypatch.setattr(SplitStats, 'subset_frac_plot', lambda self, p='': calls.append('frac'))
    monkeypatch.setattr(SplitStats, 'dist_hist_kfold_plot', lambda self, *a, **k: calls.append('kf_tvt'))
    monkeypatch.setattr(SplitStats, 'dist_hist_kfold_tvv_plot', lambda self, *a, **k: calls.append('kf_tvv'))
    monkeypatch.setattr(SplitStats, 'dist_hist_plot', lambda self, *a, **k: calls.append('std_hist'))

    kfold_ss.make_all_plots()
    assert calls == ['kf_tvt', 'kf_tvv', 'umap', 'frac']

    calls.clear()
    standard_ss.make_all_plots()
    assert calls == ['std_hist', 'std_hist', 'umap', 'frac']


# --- standard-split helpers that were uncovered -----------------------------------------------
def test_dist_hist_plot_standard_saves_figure(standard_ss, tmp_path):
    prefix = str(tmp_path / 'hist')
    standard_ss.dist_hist_plot(standard_ss.dists_tvt, 'a title', dist_path=prefix)
    assert os.path.exists(prefix + '_dist_hist.png')
    assert os.path.exists(prefix + '_dist_hist.svg')


def test_save_figure_writes_png_and_svg(tmp_path):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    save_figure(str(tmp_path / 'fig'))
    assert (tmp_path / 'fig.png').exists()
    assert (tmp_path / 'fig.svg').exists()


# --- response_cols handling -------------------------------------------------------------------
def test_response_cols_string_is_normalized_to_list(total_df, standard_split_df):
    ss = SplitStats(total_df, standard_split_df, 'smiles', 'cmpd_id', 'response')
    assert ss.response_cols == ['response']


def test_response_cols_comma_separated_string(total_df, standard_split_df):
    total_df['response2'] = total_df['response']
    ss = SplitStats(total_df, standard_split_df, 'smiles', 'cmpd_id', 'response, response2')
    assert ss.response_cols == ['response', 'response2']


def test_response_cols_tuple_is_converted_to_list(total_df, standard_split_df):
    ss = SplitStats(total_df, standard_split_df, 'smiles', 'cmpd_id', ('response',))
    assert ss.response_cols == ['response']


@pytest.mark.parametrize('bad', [42, None, {'response'}])
def test_response_cols_rejects_invalid_type(total_df, standard_split_df, bad):
    with pytest.raises(ValueError, match="response_cols must be a string or a list"):
        SplitStats(total_df, standard_split_df, 'smiles', 'cmpd_id', bad)


def test_subset_frac_plot_standard_with_string_response_cols(total_df, standard_split_df, tmp_path):
    ss = SplitStats(total_df, standard_split_df, 'smiles', 'cmpd_id', 'response')
    prefix = str(tmp_path / 'std')
    ss.subset_frac_plot(dist_path=prefix)
    assert os.path.exists(prefix + '_frac_box.png')


# --- command line entry point -----------------------------------------------------------------
def test_main_plots_both_splits(total_df, standard_split_df, kfold_split_df, tmp_path):
    csv = tmp_path / 'data.csv'
    split_a = tmp_path / 'split_a.csv'
    split_b = tmp_path / 'split_b.csv'
    out = tmp_path / 'out'
    out.mkdir()
    total_df.to_csv(csv, index=False)
    standard_split_df.to_csv(split_a, index=False)
    kfold_split_df.to_csv(split_b, index=False)

    csp.main([str(csv), 'cmpd_id', 'smiles', str(split_a), str(split_b), str(out),
              '--response_cols', 'response'])

    assert (out / 'split_a_tvt_dist_hist.png').exists()
    assert (out / 'split_b_kfold_tvt.png').exists()


def test_parse_args_requires_response_cols(capsys):
    with pytest.raises(SystemExit):
        csp.parse_args(['d.csv', 'id', 'smiles', 'a.csv', 'b.csv', 'out'])
    assert '--response_cols' in capsys.readouterr().err


def test_main_block_invokes_main(total_df, standard_split_df, kfold_split_df, tmp_path, monkeypatch):
    csv = tmp_path / 'data.csv'
    split_a = tmp_path / 'split_a.csv'
    split_b = tmp_path / 'split_b.csv'
    out = tmp_path / 'out'
    out.mkdir()
    total_df.to_csv(csv, index=False)
    standard_split_df.to_csv(split_a, index=False)
    kfold_split_df.to_csv(split_b, index=False)
    monkeypatch.setattr(sys, 'argv', ['compare_splits_plots.py', str(csv), 'cmpd_id', 'smiles',
                                      str(split_a), str(split_b), str(out),
                                      '--response_cols', 'response'])
    runpy.run_path(csp.__file__, run_name='__main__')
    assert (out / 'split_b_kfold_tvv.png').exists()
