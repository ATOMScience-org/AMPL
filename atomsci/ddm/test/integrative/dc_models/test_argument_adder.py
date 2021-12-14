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

    assert type(args.b_a_int)==float
    assert type(args.b_a_float)==float
    assert type(args.b_a_list)==str
    assert type(args.b_b_namespace)==str

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

if __name__ == '__main__':
    test_add_arguments_to_parser()
    test_list_float_args()
    test_list_int_args()
    test_list_args()
