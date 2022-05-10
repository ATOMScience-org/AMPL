import os

from atomsci.ddm.pipeline import compare_models as cm

def clean():
    pass

def test():
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

if __name__ == '__main__':
    test_graphconv()
    test()