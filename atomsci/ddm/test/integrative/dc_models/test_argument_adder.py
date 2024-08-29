import atomsci.ddm.pipeline.parameter_parser as pp
import argparse
from typing import List, Union

class A:
    def __init__(self,
        a_int: int,
        a_float: float,
        a_list: list,
        a_nother_list: List[str],
        a_union_list_float: Union[List[float], None],
        a_union_list_int: Union[List[int], None],
        a_union: Union[List[str], None]):
        print('made class A')

class B(A):
    def __init__(self,
        a_int: float,
        b_nother_int: int,
        b_list_float: List[float],
        b_namespace: argparse.Namespace,
        b_list_int: List[int],):
        print('made class B')

def test_add_arguments_to_parser():
    parser = argparse.ArgumentParser()
    aaa = pp.AutoArgumentAdder(func=B,prefix='b')
    aaa.add_to_parser(parser)

    args = parser.parse_args('--b_a_int=3.0 --b_a_float=1 --b_a_list="[1,2,3]" --b_b_namespace="Test"'.split())

    assert type(args.b_a_int) is float
    assert type(args.b_a_float) is float
    assert type(args.b_a_list) is str
    assert type(args.b_b_namespace) is str

def test_list_float_args():
    aaa = pp.AutoArgumentAdder(func=B,prefix='b')

    float_list = aaa.get_list_float_args()

    assert set(float_list) == set(['b_a_union_list_float', 'b_b_list_float'])

def test_list_int_args():
    aaa = pp.AutoArgumentAdder(func=B,prefix='b')

    int_list = aaa.get_list_int_args()

    assert set(int_list) == set(['b_a_union_list_int', 'b_b_list_int'])

def test_list_args():
    aaa = pp.AutoArgumentAdder(func=B,prefix='b')

    list_list = aaa.get_list_args()

    assert set(list_list) == set(['b_b_list_int', 'b_b_list_float',
        'b_a_list', 'b_a_nother_list', 'b_a_union_list_int', 'b_a_union_list_float', 'b_a_union'])

def test_synonyms():
    answer_a = {
        "mode": "regression",
        "num_layers": 3,
        "learning_rate": 0.0007,
        "n_tasks": 1,
    }

    answer_c = {
        "mode": "classification",
        "num_layers": 3,
        "learning_rate": 0.0007,
        "n_tasks": 2,
    }

    json_a = {
        "AttentiveFPModel_mode": "regression",
        "AttentiveFPModel_num_layers":"3",
        "AttentiveFPModel_learning_rate": "0.0007",
        "response_cols": "asdf"
    }

    json_b = {
        "prediction_type": "regression",
        "AttentiveFPModel_num_layers":"3",
        "learning_rate": "0.0007",
        "response_cols": "asdf"
    }

    json_c = {
        "prediction_type": "classification",
        "AttentiveFPModel_num_layers":"3",
        "learning_rate": "0.0007",
        "response_cols": ["asdf1", "asdf2"]
    }

    params_a = pp.wrapper(json_a)
    params_b = pp.wrapper(json_b)
    params_c = pp.wrapper(json_c)

    aaa = pp.AutoArgumentAdder(pp.model_wl['AttentiveFPModel'], 'AttentiveFPModel')

    assert aaa.extract_params(params_a) == aaa.extract_params(params_b)
    assert aaa.extract_params(params_a, strip_prefix=True) == answer_a
    assert answer_c == aaa.extract_params(params_c, strip_prefix=True)
    assert not aaa.extract_params(params_a) == aaa.extract_params(params_c)
    assert not aaa.extract_params(params_b) == aaa.extract_params(params_c)

def test_defaults():
    json_d = {
        "prediction_type": "classification",
        "AttentiveFPModel_num_layers":"3",
        "response_cols": ["asdf1", "asdf2"]
    }

    params_d = pp.wrapper(json_d)
    # make sure that the default value of synonyms are still set correctly
    expected_lr = 0.0005
    assert params_d.learning_rate == expected_lr, f'{params_d.learning_rate} should be {expected_lr}'

if __name__ == '__main__':
    test_add_arguments_to_parser()
    test_list_float_args()
    test_list_int_args()
    test_list_args()

    test_synonyms()
    test_defaults()
