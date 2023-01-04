import os
import pandas as pd

import atomsci.ddm.utils.many_to_one as mto
import atomsci.ddm.pipeline.parameter_parser as pp

def test_mto_pass():
    pass_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/many_to_one_pass.csv'))
    mto.many_to_one(pass_file, 'smiles_col', 'id_col')

def test_mto_fail():
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/many_to_one_fail.csv'))
    try:
        mto.many_to_one(fail_file, 'smiles_col', 'id_col')
        raise Exception('This should have failed')
    except mto.ManyToOneException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected ManyToOneException')

def test_mto_fail2():
    fail_df = pd.DataFrame(data={'id_col':['a', 'b', 'c', 'c', 'a', 'd'], 'smiles_col':['1', '2', '3', '4', '1', '5']})
    try:
        mto.many_to_one_df(fail_df, 'smiles_col', 'id_col')
        raise Exception('This should have failed')
    except mto.ManyToOneException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected ManyToOneException')

def test_mto_pass2():
    fail_df = pd.DataFrame(data={'id_col':['a', 'b', 'c', 'c', 'a', 'd'], 
        'smiles_col':['1', '1', '3', '3', '1', '5']})
    mto.many_to_one_df(fail_df, 'smiles_col', 'id_col')

def test_mto_nan():
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/many_to_one_nan.csv'))
    try:
        mto.many_to_one(fail_file, 'smiles_col', 'id_col')
        raise Exception('This should have failed. NANs exist in Compound ID column')
    except mto.NANCompoundIDException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected NANCompoundIDException')

def test_mto_nan2():
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/many_to_one_nan2.csv'))
    try:
        mto.many_to_one(fail_file, 'smiles_col', 'id_col')
        raise Exception('This should have failed. NANs exist in SMILES column')
    except mto.NANSMILESException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected NANSMILESException')

def test_pp_pass():
    pass_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/many_to_one_pass.csv'))
    params = {
        'split_only': 'True', 
        'splitter': 'scaffold',
        'smiles_col': 'smiles_col',
        'id_col': 'id_col', # also pull out the compound_id col from original dfs
        'response_cols': 'response_col',
        'previously_split': 'False',
        'split_valid_frac': '0.20',
        'split_test_frac': '0.20',
        "system": "LC",
        "datastore": "False",
        "save_results": "False",
        "data_owner": "username",
        "featurizer": "ecfp",
        'dataset_key': pass_file,
        'result_dir': 'results',
    }

    parsed_params = pp.wrapper(params)

def test_pp_fail():
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/many_to_one_fail.csv'))
    params = {
        'split_only': 'True', 
        'splitter': 'scaffold',
        'smiles_col': 'smiles_col',
        'id_col': 'id_col', # also pull out the compound_id col from original dfs
        'response_cols': 'response_col',
        'previously_split': 'False',
        'split_valid_frac': '0.20',
        'split_test_frac': '0.20',
        "system": "LC",
        "datastore": "False",
        "save_results": "False",
        "data_owner": "username",
        "featurizer": "ecfp",
        'dataset_key': fail_file,
        'result_dir': 'results',
    }

    try:
        parsed_params = pp.wrapper(params)
        raise Exception('This should have failed')
    except mto.ManyToOneException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected ManyToOneException')

if __name__ == '__main__':
    test_mto_pass2()
    test_pp_fail()
    test_pp_pass()
    test_mto_nan2()
    test_mto_nan()
    test_mto_pass()
    test_mto_fail()
    test_mto_fail2()
