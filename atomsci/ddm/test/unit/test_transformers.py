import atomsci.ddm.pipeline.transformations as trans
import numpy as np
from deepchem.data import NumpyDataset


def test_no_missing_values():
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    w = np.array([[1, 1], [1, 1], [1, 1]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [3.0, 4.0])
    np.testing.assert_array_almost_equal(y_stds, [1.632993, 1.632993])

def test_some_missing_values():
    y = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    w = np.array([[1, 0], [1, 1], [1, 1]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [3.0, 5.0])
    np.testing.assert_array_almost_equal(y_stds, [1.632993, 1.0])

def test_all_missing_values():
    y = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    w = np.array([[0, 0], [0, 0], [0, 0]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [0.0, 0.0])
    np.testing.assert_array_almost_equal(y_stds, [0.0, 0.0])

def test_one_task_no_missing_values():
    y = np.array([[1.0], [3.0], [5.0]])
    w = np.array([[1, 1], [1, 1], [1, 1]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [3.0])
    np.testing.assert_array_almost_equal(y_stds, [1.632993])


