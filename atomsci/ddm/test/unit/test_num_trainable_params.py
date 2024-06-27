import os

from atomsci.ddm.pipeline import compare_models as cm
from atomsci.ddm.utils import llnl_utils

def clean():
    pass

def test():
    if not llnl_utils.is_lc_system():
        assert True
        return
    
    tar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
        '../../examples/BSEP/models/bsep_classif_scaffold_split.tar.gz')
    print(tar_file)
    num_params = cm.num_trainable_parameters_from_file(tar_file)
    print(num_params)

    assert num_params == 25147

def test_graphconv():
    tar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
        '../../examples/BSEP/models/bsep_classif_scaffold_split_graphconv.tar.gz')
    print(tar_file)
    num_params = cm.num_trainable_parameters_from_file(tar_file)
    print(num_params)

    assert num_params == 194306

def test_versions_not_compatible():
    tar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '../../examples/archive/tutorials2021/models/aurka_union_trainset_base_smiles_model_ampl_120_2fb9ae80-26d4-4f27-af28-2e9e9db594de.tar.gz')
    print(tar_file)
    try:
        num_params = cm.num_trainable_parameters_from_file(tar_file)
    except ValueError as exc:
        assert True, f"Version compatible check has raised an exception {exc}"

def test_versions_compatible():
    tar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        '../../examples/BSEP/models/bsep_classif_scaffold_split_graphconv.tar.gz')
    print(tar_file)
    try:
        num_params = cm.num_trainable_parameters_from_file(tar_file)
    except ValueError as exc:
        assert False, f"Version compatible check has raised an exception {exc}"

if __name__ == '__main__':
    test_versions_not_compatible()
    test_versions_compatible()
    test_graphconv()
    test()
