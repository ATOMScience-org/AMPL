from deepchem.data import DiskDataset
import numpy as np
import pandas as pd
from atomsci.ddm.pipeline.splitting import DatasetManager, _copy_modify_NumpyDataset

def make_test_dataset_and_attr():
    ids = ['a', 'a', 'b', 'b', 'c']
    dd = DiskDataset.from_numpy(
        X=np.ones((5, 10)),
        y=np.ones((5, 4)),
        w = np.array([
            [1, 0, 0, 0,],
            [0, 1, 0, 0,],
            [0, 0, 1, 0,],
            [0, 0, 1, 0,],
            [0, 0, 0, 1,],
        ]),
        ids=ids
    )

    smiles_col = 'smiles'
    id_col = 'compound_ids'
    attr_df = pd.DataFrame({smiles_col:['aaa', 'aaa', 'bbb', 'bbb', 'ccc'], id_col:ids})
    attr_df = attr_df.set_index(id_col)

    return dd, attr_df, smiles_col

def make_test_dataset_and_attrB():
    ids = ['a', 'b', 'c', 'd', 'e']
    dd = DiskDataset.from_numpy(
        X=np.ones((5, 10)),
        y=np.ones((5, 4)),
        w = np.array([
            [1, 0, 0, 0,],
            [0, 1, 0, 0,],
            [0, 0, 1, 0,],
            [0, 0, 1, 0,],
            [0, 0, 0, 1,],
        ]),
        ids=ids
    )

    smiles_col = 'smiles'
    id_col = 'compound_ids'
    attr_df = pd.DataFrame({smiles_col:['aaa', 'bbb', 'ccc', 'ddd', 'eee'], id_col:ids})
    attr_df = attr_df.set_index(id_col)

    return dd, attr_df, smiles_col

def check_ids(expected_ids, dataset):
    assert len(expected_ids) == len(dataset)
    assert all([a==b for a, b in zip(dataset.ids, expected_ids)]), f'Expecting {expected_ids}. Instead got {dataset.ids}'

def check_ws(expected_w, dataset):
    assert np.array_equal(dataset.w, expected_w), f'Expecting {expected_w}. Instead generated {dataset.w}'

def check_arrays(a, b):
    assert np.array_equal(a, b), f'Expecting {a}. Instead generated {b}'

def check_smiles(expected_smiles, attr_df, smiles_col):
    assert len(expected_smiles) == len(attr_df)
    expanded_smiles = attr_df[smiles_col].values
    assert all([a==b for a, b in zip(expanded_smiles, expected_smiles)]), f'Expecting {expected_smiles}. Expanded {expanded_smiles}'

def test__copy_modify_NumpyDataset():
    dd, attr_df, smiles_col = make_test_dataset_and_attr()
    new_X = np.zeros_like(dd.X)
    new_y = np.zeros_like(dd.y)

    copied_dataset = _copy_modify_NumpyDataset(dd, X=new_X, y=new_y)
    check_arrays(new_X, copied_dataset.X)
    check_arrays(new_y, copied_dataset.y)

def test_DatasetManager_no_dupes():
    dd, attr_df, smiles_col = make_test_dataset_and_attrB()
    dm = DatasetManager(dataset=dd,
        attr_df=attr_df, smiles_col=smiles_col,
        needs_smiles=False)

    compact_dataset = dm.compact_dataset()
    expected_w = dd.w
    check_ws(expected_w, compact_dataset)

    expected_ids = ['a', 'b', 'c', 'd', 'e']
    check_ids(expected_ids, compact_dataset)

    selected_ids = ['a', 'b', 'e']
    sel_dataset, sel_attr = dm.expand_selection(selected_ids)
    check_ids(selected_ids, sel_dataset)

    expected_smiles = ['aaa', 'bbb', 'eee']
    check_smiles(expected_smiles, sel_attr, smiles_col)

def test_DatasetManager_doesnot_needs_smiles():
    dd, attr_df, smiles_col = make_test_dataset_and_attr()
    dm = DatasetManager(dataset=dd,
        attr_df=attr_df, smiles_col=smiles_col,
        needs_smiles=False)

    compact_dataset = dm.compact_dataset()
    expected_w = np.array([
                    [1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    check_ws(expected_w, compact_dataset)

    expected_ids = ['a', 'b', 'c']
    check_ids(expected_ids, compact_dataset)

    selected_ids = ['a', 'a', 'b', 'b']
    sel_dataset, sel_attr = dm.expand_selection(selected_ids)
    check_ids(selected_ids, sel_dataset)

    expected_smiles = ['aaa', 'aaa', 'bbb', 'bbb']
    check_smiles(expected_smiles, sel_attr, smiles_col)

def test_DatasetManager_needs_smiles():
    dd, attr_df, smiles_col = make_test_dataset_and_attr()
    dm = DatasetManager(dataset=dd,
        attr_df=attr_df, smiles_col=smiles_col,
        needs_smiles=True)

    compact_dataset = dm.compact_dataset()
    expected_w = np.array([
                    [1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    check_ws(expected_w, compact_dataset)

    expected_ids = ['aaa', 'bbb', 'ccc']
    check_ids(expected_ids, compact_dataset)

    selected_ids = ['aaa', 'bbb']
    sel_dataset, sel_attr = dm.expand_selection(selected_ids)

    expected_ids = ['a', 'a', 'b', 'b']
    check_ids(expected_ids, sel_dataset)
    expected_smiles = ['aaa', 'aaa', 'bbb', 'bbb']
    check_smiles(expected_smiles, sel_attr, smiles_col)

def test_DatasetManager_many_to_one():
    ids = ['a', 'a', 'b', 'b', 'c']
    dd = DiskDataset.from_numpy(
        X=np.ones((5, 10)),
        y=np.ones((5, 4)),
        w = np.array([
            [1, 0, 0, 0,],
            [0, 1, 0, 0,],
            [0, 0, 1, 0,],
            [0, 0, 1, 0,],
            [0, 0, 0, 1,],
        ]),
        ids=ids
    )

    smiles_col = 'smiles'
    id_col = 'compound_ids'
    attr_df = pd.DataFrame({smiles_col:['aaa', 'aaa', 'aaa', 'aaa', 'ccc'], id_col:ids})
    attr_df = attr_df.set_index(id_col)

    dm = DatasetManager(dataset=dd,
        attr_df=attr_df, smiles_col=smiles_col,
        needs_smiles=True)

    compact_dataset = dm.compact_dataset()
    expected_w = np.array([
                    [1, 1, 1, 0],
                    [0, 0, 0, 1]])
    check_ws(expected_w, compact_dataset)

    expected_ids = ['aaa', 'ccc']
    check_ids(expected_ids, compact_dataset)

    selected_ids = ['aaa']
    sel_dataset, sel_attr = dm.expand_selection(selected_ids)
    expected_ids = ['a', 'a', 'b', 'b']
    check_ids(expected_ids, sel_dataset)

    expected_smiles = ['aaa', 'aaa', 'aaa', 'aaa']
    check_smiles(expected_smiles, sel_attr, smiles_col)

if __name__ == '__main__':
    test_DatasetManager_no_dupes()
    test_DatasetManager_doesnot_needs_smiles()
    test_DatasetManager_needs_smiles()
    test_copy_modify_NumpyDataset()
    test_DatasetManager_many_to_one()
