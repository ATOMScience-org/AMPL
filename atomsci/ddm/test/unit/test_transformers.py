import atomsci.ddm.pipeline.transformations as trans
import numpy as np
from deepchem.data import NumpyDataset


def test_no_missing_values():
    """
    Test the `get_statistics_missing_ydata` function from the `trans` module
    to ensure it correctly calculates the mean and standard deviation of the
    y-values when there are no missing values in the dataset.

    The test creates a dataset with no missing y-values and checks that the
    calculated means and standard deviations match the expected values.

    Assertions:
        - The means of the y-values should be [3.0, 4.0].
        - The standard deviations of the y-values should be approximately [1.632993, 1.632993].
    """
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    w = np.array([[1, 1], [1, 1], [1, 1]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [3.0, 4.0])
    np.testing.assert_array_almost_equal(y_stds, [1.632993, 1.632993])

def test_some_missing_values():
    """
    Test the handling of missing values in the dataset.

    This test creates a dataset with some missing values in the target variable `y`
    and verifies that the `get_statistics_missing_ydata` function correctly computes
    the means and standard deviations of the non-missing values.

    The test checks that the computed means and standard deviations of the non-missing
    values in `y` match the expected values.

    Assertions:
    - The means of the non-missing values in `y` should be approximately [3.0, 5.0].
    - The standard deviations of the non-missing values in `y` should be approximately [1.632993, 1.0].
    """
    y = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    w = np.array([[1, 0], [1, 1], [1, 1]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [3.0, 5.0])
    np.testing.assert_array_almost_equal(y_stds, [1.632993, 1.0])

def test_all_missing_values():
    """
    Test the `get_statistics_missing_ydata` function with a dataset where all y-values are missing (NaN).

    This test creates a dataset with all missing y-values and checks if the function correctly computes
    the means and standard deviations of the y-values, which should both be arrays of zeros.

    The test asserts that:
    - The means of the y-values are [0.0, 0.0].
    - The standard deviations of the y-values are [0.0, 0.0].
    """
    y = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    w = np.array([[0, 0], [0, 0], [0, 0]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [0.0, 0.0])
    np.testing.assert_array_almost_equal(y_stds, [0.0, 0.0])

def test_one_task_no_missing_values():
    """
    Test the `get_statistics_missing_ydata` function with a dataset that has no missing values.

    This test creates a dataset with no missing values and checks if the mean and standard deviation
    of the y-values are calculated correctly.
    
    The expected mean of y-values is [3.0] and the expected standard deviation is [1.632993].

    Asserts:
        - The calculated mean of y-values is almost equal to [3.0].
        - The calculated standard deviation of y-values is almost equal to [1.632993].
    """
    y = np.array([[1.0], [3.0], [5.0]])
    w = np.array([[1], [1], [1]])
    x = np.ones_like(y)
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=x, y=y, w=w, ids=ids)
    y_means, y_stds = trans.get_statistics_missing_ydata(dataset)
    np.testing.assert_array_almost_equal(y_means, [3.0])
    np.testing.assert_array_almost_equal(y_stds, [1.632993])

def test_normalization_transformer_missing_data():
    """
    Test the NormalizationTransformerMissingData class for handling missing data in the target variable.

    The expected means and standard deviations for `y` are:
    - Means: [3.0, 5.0]
    - Standard deviations: [1.632993, 1.0]

    The expected transformed `y` values are:
    - [[-1.224745, 0], [0.0, -1.0], [1.224745, 1.0]]
    """
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
    """
    Test the NormalizationTransformerMissingData with transform_X=True.

    This test verifies the following:
    1. The means and standard deviations of the features in the dataset are correctly computed.
    2. The transformation is correctly applied to the dataset.
    
    Assertions:
    - The computed means of the features should be [3.0, 4.0].
    - The computed standard deviations of the features should be approximately [1.632993, 1.632993].
    - The transformed feature values should match the expected transformed values with a precision of 6 decimal places.
    """
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

