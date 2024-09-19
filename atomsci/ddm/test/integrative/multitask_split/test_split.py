import pandas as pd
import numpy as np
import json
import glob
import os
import shutil
from deepchem.splits import ScaffoldSplitter

from atomsci.ddm.pipeline.MultitaskScaffoldSplit import MultitaskScaffoldSplitter, split_with
from atomsci.ddm.utils.compare_splits_plots import SplitStats
import atomsci.ddm.pipeline.parameter_parser as parse
from atomsci.ddm.pipeline.model_pipeline import ModelPipeline

def init_data():
    """Copy files necessary for running tests"""
    df = pd.read_csv('../../test_datasets/KCNA5_KCNH2_SCN5A_data.csv')
    df['base_rdkit_smiles'] = df['compound_id']
    df.to_csv('KCNA5_KCNH2_SCN5A_data.csv')

    if not os.path.exists('plots'):
        os.makedirs('plots')

def delete_file(file):
    """checks if a file exists and deletes it"""
    if os.path.exists(file):
        os.remove(file)

def clean():
    """cleans up files created during the test"""
    if os.path.exists('plots'):
        shutil.rmtree('plots')

    if os.path.exists('result'):
        shutil.rmtree('result')

    files = glob.glob('KCNA5_KCNH2_SCN5A_data*')
    for f in files:
        # should remove KCNA5_KCNH2_SCN5A_data.csv and split
        delete_file(f)
    delete_file('one_gen_split.csv')
    delete_file('thirty_gen_split.csv')
    delete_file('ss_split.csv')

def test_seeded_splits():
    clean()

    init_data()

    smiles_col = 'compound_id'
    id_col = 'compound_id'
    frac_train = 0.8
    frac_test = 0.1
    frac_valid = 0.1
    num_super_scaffolds = 60
    dfw = 2 # chemical distance importance weight
    rfw = 1 # split fraction importance weight

    total_df = pd.read_csv('KCNA5_KCNH2_SCN5A_data.csv', dtype={id_col:str})
    response_cols = ['target_KCNA5_standard_value',
        'target_KCNH2_standard_value',
        'target_SCN5A_activity']

    # -------------------------------------------------------------------------
    # one generation multitask scaffold split
    mss = MultitaskScaffoldSplitter()
    A_split_df = split_with(total_df, mss, 
        smiles_col=smiles_col, id_col=id_col, response_cols=response_cols, 
        diff_fitness_weight_tvt=dfw, ratio_fitness_weight=rfw, num_generations=1,
        num_super_scaffolds=num_super_scaffolds,
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid, seed=0)

    b_mss = MultitaskScaffoldSplitter()
    B_split_df = split_with(total_df, b_mss, 
        smiles_col=smiles_col, id_col=id_col, response_cols=response_cols, 
        diff_fitness_weight_tvt=dfw, ratio_fitness_weight=rfw, num_generations=1,
        num_super_scaffolds=num_super_scaffolds,
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid, seed=0)

    c_mss = MultitaskScaffoldSplitter()
    C_split_df = split_with(total_df, c_mss, 
        smiles_col=smiles_col, id_col=id_col, response_cols=response_cols, 
        diff_fitness_weight_tvt=dfw, ratio_fitness_weight=rfw, num_generations=1,
        num_super_scaffolds=num_super_scaffolds,
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid, seed=42)

    assert all(A_split_df['cmpd_id']==B_split_df['cmpd_id']) and all(A_split_df['subset']==B_split_df['subset'])
    # compounds can be in the same order
    assert not all(A_split_df['subset']==C_split_df['subset'])

    clean()

def test_splits():
    clean()

    init_data()

    smiles_col = 'compound_id'
    id_col = 'compound_id'
    output_dir = 'plots'
    frac_train = 0.8
    frac_test = 0.1
    frac_valid = 0.1
    num_super_scaffolds = 60
    num_generations = 30
    dfw = 2 # chemical distance importance weight
    rfw = 1 # split fraction importance weight

    total_df = pd.read_csv('KCNA5_KCNH2_SCN5A_data.csv', dtype={id_col:str})
    response_cols = ['target_KCNA5_standard_value',
        'target_KCNH2_standard_value',
        'target_SCN5A_activity']

    # -------------------------------------------------------------------------
    # one generation multitask scaffold split
    mss = MultitaskScaffoldSplitter()
    mss_split_df = split_with(total_df, mss, 
        smiles_col=smiles_col, id_col=id_col, response_cols=response_cols, 
        diff_fitness_weight_tvt=dfw, ratio_fitness_weight=rfw, num_generations=1,
        num_super_scaffolds=num_super_scaffolds,
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid, seed=0)
    mss_split_df.to_csv('one_gen_split.csv', index=False)
    assert len(total_df) == len(mss_split_df)

    split_a = pd.read_csv('one_gen_split.csv', dtype={'cmpd_id':str})
    split_a_ss = SplitStats(total_df, split_a, smiles_col=smiles_col, 
        id_col=id_col, response_cols=response_cols)
    split_a_ss.make_all_plots(dist_path=os.path.join(output_dir, 'multitask_1gen'))

    # -------------------------------------------------------------------------
    # multiple generation mulittask scaffold split
    mss = MultitaskScaffoldSplitter()
    mss_split_df = split_with(total_df, mss, 
        smiles_col=smiles_col, id_col=id_col, response_cols=response_cols, 
        diff_fitness_weight_tvt=dfw, ratio_fitness_weight=rfw,
        num_generations=num_generations,
        num_super_scaffolds=num_super_scaffolds,
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid, seed=0)
    mss_split_df.to_csv('thirty_gen_split.csv', index=False)
    assert len(total_df) == len(mss_split_df)

    split_b = pd.read_csv('thirty_gen_split.csv', dtype={'cmpd_id':str})
    split_b_ss = SplitStats(total_df, split_b, smiles_col=smiles_col, 
        id_col=id_col, response_cols=response_cols)
    split_b_ss.make_all_plots(dist_path=os.path.join(output_dir, 
        f'multitask_{num_generations}gen'))

    # -------------------------------------------------------------------------
    # regular scaffold split
    ss = ScaffoldSplitter()
    ss_split_df = split_with(total_df, ss, 
        smiles_col=smiles_col, id_col=id_col, response_cols=response_cols, 
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid)
    ss_split_df.to_csv('ss_split.csv', index=False)
    assert len(total_df) == len(ss_split_df)

    split_c = pd.read_csv('ss_split.csv', dtype={'cmpd_id':str})
    split_c_ss = SplitStats(total_df, split_c, smiles_col=smiles_col, 
        id_col=id_col, response_cols=response_cols)
    split_c_ss.make_all_plots(dist_path=os.path.join(output_dir, 'scaffold_'))

    # median train/test compound distance should have gone up
    assert np.median(split_a_ss.dists_tvt) <= np.median(split_b_ss.dists_tvt)
    assert np.median(split_c_ss.dists_tvt) <= np.median(split_b_ss.dists_tvt)

    # no subset should contain 0 samples
    assert np.min(np.concatenate(
        [split_a_ss.train_fracs, split_a_ss.valid_fracs, split_a_ss.test_fracs])) > 0
    assert np.min(np.concatenate(
        [split_b_ss.train_fracs, split_b_ss.valid_fracs, split_b_ss.test_fracs])) > 0

    clean()

def test_pipeline_split_only():
    clean()
    init_data()
    json_file = 'multitask_config_split_only.json'
    params = parse.wrapper(['--config', json_file])
    mp = ModelPipeline(params)
    split_uuid = mp.split_dataset()
    # check that mp.splitter.num_super_scaffolds is right etc
    with open(json_file, 'r') as jf:
        js = json.load(jf)
    assert mp.data.splitting.splitter.num_super_scaffolds == js['mtss_num_super_scaffolds']
    assert mp.data.splitting.splitter.num_generations == js['mtss_num_generations']
    assert mp.data.splitting.splitter.num_pop == js['mtss_num_pop']
    assert mp.data.splitting.splitter.diff_fitness_weight_tvt == js['mtss_train_test_dist_weight']
    assert mp.data.splitting.splitter.diff_fitness_weight_tvv == js['mtss_train_valid_dist_weight']
    assert mp.data.splitting.splitter.ratio_fitness_weight == js['mtss_split_fraction_weight']

    files = glob.glob('KCNA5_KCNH2_SCN5A_data*')
    assert any([split_uuid in f for f in files])

    clean()

def test_pipeline_split_and_train():
    clean()
    init_data()
    json_file = 'multitask_config.json'
    params = parse.wrapper(['--config', json_file])
    mp = ModelPipeline(params)
    mp.train_model()
    with open(json_file, 'r') as jf:
        js = json.load(jf)
    assert mp.data.splitting.splitter.num_super_scaffolds == js['mtss_num_super_scaffolds']
    assert mp.data.splitting.splitter.num_generations == js['mtss_num_generations']
    assert mp.data.splitting.splitter.num_pop == js['mtss_num_pop']
    assert mp.data.splitting.splitter.diff_fitness_weight_tvt == js['mtss_train_test_dist_weight']
    assert mp.data.splitting.splitter.diff_fitness_weight_tvv == js['mtss_train_valid_dist_weight']
    assert mp.data.splitting.splitter.ratio_fitness_weight == js['mtss_split_fraction_weight']

    files = glob.glob('KCNA5_KCNH2_SCN5A_data*')
    assert any([mp.data.split_uuid in f for f in files])

    clean()

if __name__ == '__main__':
    test_seeded_splits()
    #test_splits()
    #test_pipeline_split_only()
    #test_pipeline_split_and_train()