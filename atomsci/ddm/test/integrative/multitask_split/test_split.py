import pandas as pd
import numpy as np
import os
import shutil
from deepchem.splits import ScaffoldSplitter

from atomsci.ddm.pipeline.MultitaskScaffoldSplit import MultitaskScaffoldSplitter, split_with
from atomsci.ddm.utils.compare_splits_plots import SplitStats

def init_data():
    '''
    Copy files necessary for running tests
    '''
    shutil.copyfile('../../test_datasets/KCNA5_KCNH2_SCN5A_data.csv', 'KCNA5_KCNH2_SCN5A_data.csv')

    if not os.path.exists('plots'):
        os.makedirs('plots')

def delete_file(file):
    '''
    checks if a file exists and deletes it
    '''
    if os.path.exists(file):
        os.remove(file)

def clean():
    '''
    cleans up files created during the test
    '''
    if os.path.exists('plots'):
        shutil.rmtree('plots')

    delete_file('KCNA5_KCNH2_SCN5A_data.csv')
    delete_file('one_gen_split.csv')
    delete_file('twenty_gen_split.csv')
    delete_file('ss_split.csv')

def test_splits():
    clean()

    init_data()

    smiles_col = 'compound_id'
    id_col = 'compound_id'
    output_dir = 'plots'
    frac_train = 0.8
    frac_test = 0.1
    frac_valid = 0.1
    num_super_scaffolds = 40
    num_generations = 20
    dfw = 1 # chemical distance importance weight
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
        diff_fitness_weight=dfw, ratio_fitness_weight=rfw, num_generations=1,
        num_super_scaffolds=num_super_scaffolds,
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid)
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
        diff_fitness_weight=dfw, ratio_fitness_weight=rfw,
        num_generations=num_generations,
        num_super_scaffolds=num_super_scaffolds,
        frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid)
    mss_split_df.to_csv('twenty_gen_split.csv', index=False)
    assert len(total_df) == len(mss_split_df)

    split_b = pd.read_csv('twenty_gen_split.csv', dtype={'cmpd_id':str})
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
    assert np.median(split_a_ss.dists) <= np.median(split_b_ss.dists)
    assert np.median(split_c_ss.dists) <= np.median(split_b_ss.dists)

    # no subset should contain 0 samples
    assert np.min(np.concatenate(
        [split_a_ss.train_fracs, split_a_ss.valid_fracs, split_a_ss.test_fracs])) > 0
    assert np.min(np.concatenate(
        [split_b_ss.train_fracs, split_b_ss.valid_fracs, split_b_ss.test_fracs])) > 0

    clean()

if __name__ == '__main__':
    test_splits()