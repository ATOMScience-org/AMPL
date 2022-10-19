import atomsci.ddm.utils.one_to_one as oto
import atomsci.ddm.pipeline.parameter_parser as pp
import os

def test_oto_pass():
    pass_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/one_to_one_pass.csv'))
    oto.one_to_one(pass_file, 'smiles_col', 'id_col')

def test_oto_fail():
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/one_to_one_fail.csv'))
    try:
        oto.one_to_one(fail_file, 'smiles_col', 'id_col')
        raise Exception('This should have failed')
    except oto.OneToOneException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected OneToOneException')

def test_oto_nan():
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/one_to_one_nan.csv'))
    try:
        oto.one_to_one(fail_file, 'smiles_col', 'id_col')
        raise Exception('This should have failed. NANs exist in Compound ID column')
    except oto.NANCompoundIDException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected NANCompoundIDException')

def test_oto_nan2():
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/one_to_one_nan2.csv'))
    try:
        oto.one_to_one(fail_file, 'smiles_col', 'id_col')
        raise Exception('This should have failed. NANs exist in SMILES column')
    except oto.NANSMILESException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected NANSMILESException')

def test_pp_pass():
    pass_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/one_to_one_pass.csv'))
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
    fail_file = os.path.abspath(os.path.join(__file__, '../../test_datasets/one_to_one_fail.csv'))
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
    except oto.OneToOneException:
        # we expect this error
        pass
    except:
        raise Exception('Got the wrong exception, expected OneToOneException')



if __name__ == '__main__':
    test_pp_fail()
    test_pp_pass()
    test_oto_nan2()
    test_oto_nan()
    test_oto_pass()
    test_oto_fail()