#!/usr/bin/env python

import pandas as pd
import os

import pytest

import atomsci.ddm.utils.curate_data as curate_data

script_path = os.path.dirname(os.path.realpath(__file__))
test_file_prefix = 'pGP_MDCK_efflux_ratio_chembl29'
test_files = [f"{script_path}/{test_file_prefix}-{suffix}.csv" for suffix in ['filtered', 'aggregated', 'averaged']]
filt_file = f"{script_path}/{test_file_prefix}-filtered.csv"
filt_df = None

def clean():
    """
    Clean test files
    """
    for f in test_files:
        if os.path.isfile(f):
            os.remove(f)

def get_raw_data():
    """Returns data frame for dataset to be used to test curation functions"""
    dset_path = os.path.realpath(os.path.join(script_path, 
        '../../test_datasets/pGP_MDCK_efflux_ratio_chembl29.csv'))
    raw_df = pd.read_csv(dset_path)
    return raw_df

def write_to_file(filt_df, filt_file):
    filt_df.to_csv(filt_file, index=False)
    print(f"Wrote outlier-filtered data to {filt_file}")
    return filt_df

def create_raw_and_filt_file():
    """Create filtered file for testing aggregation function"""
    raw_df = get_raw_data()
    filt_df = curate_data.remove_outlier_replicates(raw_df, response_col='log_efflux_ratio', id_col='base_rdkit_smiles',
                                                    max_diff_from_median=0.5)
    write_to_file(filt_df, filt_file)

    return raw_df, filt_df

def test_remove_outlier_replicates(capsys):
    """Test outlier removal using curate_data.remove_outlier_replicates"""
     # Clean up old files
    clean()

    raw_df, filt_df = create_raw_and_filt_file()

    captured = capsys.readouterr()
    assert 'Removed 1 rows with missing log_efflux_ratio values' in captured.out, "Error: expected message about removed rows with missing values"
    n_filt_rows = len(filt_df)
    n_filt_cmpds = len(set(filt_df.base_rdkit_smiles.values))
    print(f"Filtered data has {n_filt_rows} rows, {n_filt_cmpds} unique compounds")
    assert (n_filt_rows == 1093), "Error: expected 1093 rows in filtered data"
    assert (n_filt_cmpds == 803), "Error: expected 803 unique compounds in filtered data"
    n_removed = len(raw_df) - n_filt_rows
    assert (n_removed == 8), f"Error: {n_removed} rows were removed, expected 8"

def test_aggregate_assay_data():
    """Test curate_data.aggregate_assay_data, the preferred function for averaging replicate values over compounds"""
     # Clean up old files
    clean()
    raw_df, filt_df = create_raw_and_filt_file()
    agg_df = curate_data.aggregate_assay_data(filt_df, value_col='log_efflux_ratio', label_actives=False,
                                        id_col='compound_id', smiles_col='base_rdkit_smiles', relation_col='relation')
    n_agg_rows = len(agg_df)
    n_agg_cmpds = len(set(agg_df.base_rdkit_smiles.values))
    print(f"Aggregated data has {n_agg_rows} rows, {n_agg_cmpds} unique compounds")
    assert (n_agg_rows == 803), "Error: expected 803 rows in aggregated data"
    assert (n_agg_cmpds == 803), "Error: expected 803 unique compounds in aggregated data"

    agg_file = f"{script_path}/{test_file_prefix}-aggregated.csv"
    agg_df.to_csv(agg_file, index=False)
    print(f"Wrote aggregated data to {agg_file}")

def test_average_and_remove_duplicates():
    """Test outlier removal and averaging using deprecated curation function"""
    raw_df = get_raw_data()

    # tolerance: In each iteration, remove replicate measurements that differ from their mean by more than this percentage of the mean
    tolerance = 50  # percentage
    column = 'log_efflux_ratio' # column containing measurement values
    list_bad_duplicates = 'Yes'
    data = raw_df
    # max_std: Remove compounds whose standard deviation across replicates exceeds this value
    max_std = 0.5

    curated_df = curate_data.average_and_remove_duplicates(
        column, tolerance, list_bad_duplicates, data, max_std, compound_id='compound_id', smiles_col='base_rdkit_smiles')
    print(f"Averaged data has {len(curated_df)} rows, {len(set(curated_df.base_rdkit_smiles.values))} unique compounds")

    curated_file = f"{script_path}/{test_file_prefix}-curated.csv"
    curated_df.to_csv(curated_file, index=False)
    print(f"Wrote curated data to {curated_file}")

