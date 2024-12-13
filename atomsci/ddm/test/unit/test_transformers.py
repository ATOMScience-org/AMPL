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
    w = np.array([[1], [1], [1]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [3.0])
    np.testing.assert_array_almost_equal(y_stds, [1.632993])

def test_normalization_transformer_missing_data():
    # Create a mock dataset
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    w = np.array([[1, 0], [1, 1], [1, 1]])
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=X, y=y, w=w, ids=ids)

    # Initialize the transformer
    transformer = trans.NormalizationTransformerMissingData(transform_X=False, transform_y=True, dataset=dataset)

    # Check the means and standard deviations
    expected_y_means = np.array([3.0, 5.0])
    expected_y_stds = np.array([1.632993, 1.0])
    np.testing.assert_array_almost_equal(transformer.y_means, expected_y_means)
    np.testing.assert_array_almost_equal(transformer.y_stds, expected_y_stds)

    # Apply the transformation
    transformed_dataset = transformer.transform(dataset)

    # Check the transformed values
    # np.nan is replaced with 0
    expected_transformed_y = np.array([[-1.224745, 0], [0.0, -1.0], [1.224745, 1.0]])
    np.testing.assert_array_almost_equal(transformed_dataset.y, expected_transformed_y, decimal=6)

def test_normalization_transformer_missing_data_transform_X():
    # Create a mock dataset
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([[1.0], [3.0], [5.0]])
    w = np.array([[1], [1], [1]])
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=X, y=y, w=w, ids=ids)

    # Initialize the transformer with transform_X=True
    transformer = trans.NormalizationTransformerMissingData(transform_X=True, dataset=dataset)

    # Check the means and standard deviations
    expected_X_means = np.array([3.0, 4.0])
    expected_X_stds = np.array([1.632993, 1.632993])
    np.testing.assert_array_almost_equal(transformer.X_means, expected_X_means)
    np.testing.assert_array_almost_equal(transformer.X_stds, expected_X_stds)

    # Apply the transformation
    transformed_dataset = transformer.transform(dataset)

    # Check the transformed values
    expected_transformed_X = (X - expected_X_means) / expected_X_stds
    np.testing.assert_array_almost_equal(transformed_dataset.X, expected_transformed_X, decimal=6)

