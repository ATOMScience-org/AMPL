#!/usr/bin/env python

import pandas as pd
import os
import sys
import shutil

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.compare_models as cm
from atomsci.ddm.utils.query_by_committee import (
    query_by_committee_regression,
    plot_qbc_overview,
    select_high_pred_high_certainty,
    select_high_pred_medium_certainty,
    select_diverse_subset,
    QBCColumns,
)
from atomsci.ddm.utils.query_by_committee_utils import (
    add_murcko_scaffolds,
    cluster_by_scaffold,
    scaffold_summary,
    pick_one_per_scaffold,
    scorer_most_active,
    scorer_pred_minus_k_std,
    plot_scaffold_prediction_uncertainty,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import integrative_utilities


def clean():
    """Clean test files"""
    for f in ['MRP3_curated.csv']:
        if os.path.isfile(f):
            os.remove(f)
    if os.path.exists('result'):
        shutil.rmtree('result')
    if os.path.exists('predict'):
        shutil.rmtree('predict')


def train_models(num_models=5):
    """Train multiple RF models with different seeds"""
    model_paths = []
    base_result_dir = 'result/mrp3_qbc'

    # Check if there are already enough trained models
    results_df = cm.get_filesystem_perf_results(base_result_dir,
                                                  pred_type='regression')
    if len(results_df) > num_models:
        model_paths = results_df['model_path'].values[:num_models]
    else:
        num_needed_models = num_models-len(results_df)

        for i in range(num_needed_models):
            seed = str(i * 42)
            result_dir = f'{base_result_dir}_seed_{seed}'

            params = {
                "prediction_type": "regression",
                "dataset_key": "../../test_datasets/MRP3_dataset.csv",
                "id_col": "compound_id",
                "smiles_col": "rdkit_smiles",
                "response_cols": "pIC50",
                "previously_split": "False",
                "splitter": "random",
                "split_valid_frac": "0.15",
                "split_test_frac": "0.15",
                "featurizer": "computed_descriptors",
                "descriptor_type": "rdkit_raw",
                "model_type": "RF",
                "seed": seed,
                "transformers": "True",
                "result_dir": result_dir,
                "rf_estimators": "100",
            }

            ampl_param = parse.wrapper(params)
            pl = mp.ModelPipeline(ampl_param)
            pl.train_model()

        results_df = cm.get_filesystem_perf_results(base_result_dir,
                                                    pred_type='regression')
        model_paths = results_df['model_path'].values[:num_models]

    return model_paths


def load_test_data():
    """Load test data and prepare input dataframes for QBC"""
    test_df = pd.read_csv('../../test_datasets/MRP3_dataset.csv')

    raw_df = test_df[['compound_id', 'rdkit_smiles', 'pIC50']].copy()
    raw_df = raw_df.rename(columns={'pIC50': 'pIC50_actual'})

    desc_path = os.path.join(
        os.path.dirname(__file__),
        '../../test_datasets/scaled_descriptors/MRP3_dataset_with_rdkit_raw_descriptors.csv'
    )
    desc_df = pd.read_csv(desc_path)
    desc_df = desc_df[['compound_id'] + [c for c in desc_df.columns if c != 'compound_id']].copy()

    common_ids = set(raw_df['compound_id']) & set(desc_df['compound_id'])
    raw_df = raw_df[raw_df['compound_id'].isin(common_ids)].copy()
    desc_df = desc_df[desc_df['compound_id'].isin(common_ids)].copy()

    input_dfs = {
        'raw': raw_df,
        'rdkit_raw': desc_df,
    }

    return input_dfs, raw_df


def test_query_by_committee_regression():
    """Test the main query_by_committee_regression function"""
    print("Testing query_by_committee_regression...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(5)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
        dedupe='first',
        disagreement_metric='committee_std',
        sort_desc=True,
        return_long=False,
    )

    assert 'committee_mean_pred' in wide_df.columns
    assert 'committee_std' in wide_df.columns
    assert 'committee_range' in wide_df.columns
    assert 'committee_n_models' in wide_df.columns
    assert 'committee_mean_model_std' in wide_df.columns
    assert len(wide_df) > 0

    wide_df2, long_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
        return_long=True,
    )

    assert 'model_key' in long_df.columns
    assert 'pred' in long_df.columns
    assert len(long_df) > 0

    print(f"  Wide shape: {wide_df.shape}")
    print(f"  Long shape: {long_df.shape}")
    print("  query_by_committee_regression PASSED")


def test_add_murcko_scaffolds():
    """Test add_murcko_scaffolds function"""
    print("Testing add_murcko_scaffolds...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    df_with_scaffold = add_murcko_scaffolds(wide_df, smiles_col='rdkit_smiles')
    assert 'murcko_scaffold' in df_with_scaffold.columns
    assert df_with_scaffold['murcko_scaffold'].notna().sum() > 0

    df_generic = add_murcko_scaffolds(wide_df, smiles_col='rdkit_smiles', generic=True)
    assert 'murcko_scaffold' in df_generic.columns

    print("  add_murcko_scaffolds PASSED")


def test_cluster_by_scaffold():
    """Test cluster_by_scaffold function"""
    print("Testing cluster_by_scaffold...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    wide_df = add_murcko_scaffolds(wide_df, smiles_col='rdkit_smiles')
    clusters = cluster_by_scaffold(wide_df, scaffold_col='murcko_scaffold', id_col='compound_id')

    assert isinstance(clusters, dict)
    assert len(clusters) > 0
    for scaffold, group in clusters.items():
        assert 'compound_id' in group.columns
        assert len(group) > 0

    print(f"  Number of scaffold clusters: {len(clusters)}")
    print("  cluster_by_scaffold PASSED")


def test_scaffold_summary():
    """Test scaffold_summary function"""
    print("Testing scaffold_summary...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    wide_df = add_murcko_scaffolds(wide_df, smiles_col='rdkit_smiles')
    summary = scaffold_summary(wide_df, scaffold_col='murcko_scaffold', pred_col='committee_mean_pred', std_col='committee_std')

    assert 'murcko_scaffold' in summary.columns
    assert 'n_compounds' in summary.columns
    assert 'committee_mean_pred_max' in summary.columns or 'committee_mean_pred_mean' in summary.columns
    assert len(summary) > 0

    print(f"  Summary shape: {summary.shape}")
    print("  scaffold_summary PASSED")


def test_pick_one_per_scaffold():
    """Test pick_one_per_scaffold function"""
    print("Testing pick_one_per_scaffold...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    wide_df = add_murcko_scaffolds(wide_df, smiles_col='rdkit_smiles')

    picked = pick_one_per_scaffold(
        wide_df,
        scorer=scorer_most_active,
        scaffold_col='murcko_scaffold',
        id_col='compound_id',
    )

    assert 'scaffold_selection_score' in picked.columns
    assert len(picked) > 0
    assert len(picked) == picked['murcko_scaffold'].nunique()

    picked2 = pick_one_per_scaffold(
        wide_df,
        scorer=lambda df: scorer_pred_minus_k_std(df, k=1.0),
        scaffold_col='murcko_scaffold',
        id_col='compound_id',
    )

    assert len(picked2) > 0

    print(f"  Picked (most active): {len(picked)}")
    print(f"  Picked (pred - std): {len(picked2)}")
    print("  pick_one_per_scaffold PASSED")


def test_scorer_most_active():
    """Test scorer_most_active function"""
    print("Testing scorer_most_active...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    scores = scorer_most_active(wide_df, pred_col='committee_mean_pred')
    assert len(scores) == len(wide_df)
    assert scores.notna().sum() > 0

    print("  scorer_most_active PASSED")


def test_scorer_pred_minus_k_std():
    """Test scorer_pred_minus_k_std function"""
    print("Testing scorer_pred_minus_k_std...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    scores = scorer_pred_minus_k_std(wide_df, pred_col='committee_mean_pred', std_col='committee_std', k=1.0)
    assert len(scores) == len(wide_df)
    assert scores.notna().sum() > 0

    scores2 = scorer_pred_minus_k_std(wide_df, pred_col='committee_mean_pred', std_col='committee_std', k=2.0)
    assert len(scores2) == len(wide_df)

    print("  scorer_pred_minus_k_std PASSED")


def test_select_high_pred_high_certainty():
    """Test select_high_pred_high_certainty function"""
    print("Testing select_high_pred_high_certainty...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    selected = select_high_pred_high_certainty(
        wide_df,
        n=10,
        cols=QBCColumns(),
        id_col='compound_id',
        uncertainty_mode='combined',
        pred_thr=-10,
        unc_max_quantile=1.0,
    )

    assert 'selection_score' in selected.columns
    assert len(selected) <= 10

    print(f"  Selected: {len(selected)}")
    print("  select_high_pred_high_certainty PASSED")


def test_select_high_pred_medium_certainty():
    """Test select_high_pred_medium_certainty function"""
    print("Testing select_high_pred_medium_certainty...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    selected = select_high_pred_medium_certainty(
        wide_df,
        n=10,
        cols=QBCColumns(),
        id_col='compound_id',
        uncertainty_mode='combined',
        pred_min_quantile=0.5,
        unc_band=(0.0, 1.0),
    )

    assert 'selection_score' in selected.columns
    assert len(selected) <= 10

    print(f"  Selected: {len(selected)}")
    print("  select_high_pred_medium_certainty PASSED")


def test_select_diverse_subset():
    """Test select_diverse_subset function"""
    print("Testing select_diverse_subset...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    wide_df = add_murcko_scaffolds(wide_df, smiles_col='rdkit_smiles')
    wide_df['selection_score'] = wide_df['committee_mean_pred']

    selected = select_diverse_subset(
        wide_df,
        n=10,
        smiles_col='rdkit_smiles',
        score_col='selection_score',
        id_col='compound_id',
        max_candidates=1000,
        max_tanimoto=0.4,
    )

    assert len(selected) <= 10
    assert 'diversity_max_tanimoto_used' in selected.columns

    print(f"  Selected diverse: {len(selected)}")
    print("  select_diverse_subset PASSED")


def test_plot_qbc_overview():
    """Test plot_qbc_overview function"""
    print("Testing plot_qbc_overview...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    fig = plot_qbc_overview(wide_df, cols=QBCColumns())
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)

    print("  plot_qbc_overview PASSED")


def test_plot_scaffold_prediction_uncertainty():
    """Test plot_scaffold_prediction_uncertainty function"""
    print("Testing plot_scaffold_prediction_uncertainty...")

    input_dfs, raw_df = load_test_data()
    model_paths = train_models(3)

    wide_df = query_by_committee_regression(
        model_paths=model_paths,
        input_dfs=input_dfs,
        id_col='compound_id',
        smiles_col='rdkit_smiles',
        predicted_response='pIC50',
        base='intersection',
    )

    wide_df = add_murcko_scaffolds(wide_df, smiles_col='rdkit_smiles')

    fig = plot_scaffold_prediction_uncertainty(
        wide_df,
        pred_col='committee_mean_pred',
        std_col='committee_std',
        scaffold_col='murcko_scaffold',
    )
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)

    print("  plot_scaffold_prediction_uncertainty PASSED")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Query by Committee Tests")
    print("=" * 60)

    integrative_utilities.clean_fit_predict()
    clean()

    try:
        test_query_by_committee_regression()
        test_add_murcko_scaffolds()
        test_cluster_by_scaffold()
        test_scaffold_summary()
        test_pick_one_per_scaffold()
        test_scorer_most_active()
        test_scorer_pred_minus_k_std()
        test_select_high_pred_high_certainty()
        test_select_high_pred_medium_certainty()
        test_select_diverse_subset()
        test_plot_qbc_overview()
        test_plot_scaffold_prediction_uncertainty()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    finally:
        clean()
        integrative_utilities.clean_fit_predict()


if __name__ == '__main__':
    run_all_tests()
