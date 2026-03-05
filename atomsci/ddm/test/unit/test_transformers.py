import atomsci.ddm.pipeline.transformations as trans
import atomsci.ddm.pipeline.parameter_parser as pp
import numpy as np
import pandas as pd
from deepchem.data import NumpyDataset
import pytest

from sklearn.preprocessing import RobustScaler, PowerTransformer

def test_sklearn_pipeline_wrapper():
    """
    Creates a mock dataset.
        Tests the SklearnTransformerWrapper with RobustScaler on X.
        Tests the SklearnTransformerWrapper with PowerTransformer on y.
        Tests the SklearnTransformerWrapper with RobustScaler on w.
        Asserts that the transformed values match the expected values.
    """
    # Create a mock dataset
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([[1.0], [3.0], [5.0]])
    w = np.array([[1.0], [1.0], [1.0]])
    ids = np.array(range(len(y)))
    dataset = NumpyDataset(X=X, y=y, w=w, ids=ids)

    # Test with RobustScaler on X
    scaler = RobustScaler()
    transformer = trans.SklearnPipelineWrapper(dataset, scaler, transform_X=True)
    transformed_dataset = transformer.transform(dataset)
    expected_transformed_X = scaler.fit_transform(X)
    np.testing.assert_array_almost_equal(transformed_dataset.X, expected_transformed_X)

    # Test untransform on X
    with pytest.raises(NotImplementedError) as exception_info:
        untransformed_X = transformer.untransform(transformed_dataset.X)
    assert str(exception_info.value) == 'SklearnPipelineWrapper does not support inverse transforms'

    # Test with PowerTransformer on y
    power_transformer = PowerTransformer()
    transformer = trans.SklearnPipelineWrapper(dataset, power_transformer, transform_y=True)
    transformed_dataset = transformer.transform(dataset)
    expected_transformed_y = power_transformer.fit_transform(y)
    np.testing.assert_array_almost_equal(transformed_dataset.y, expected_transformed_y)

    # Test untransform on y
    with pytest.raises(NotImplementedError) as exception_info:
        untransformed_y = transformer.untransform(transformed_dataset.y)
    assert str(exception_info.value) == 'SklearnPipelineWrapper does not support inverse transforms'

    # Test with RobustScaler on w
    transformer = trans.SklearnPipelineWrapper(dataset, scaler, transform_w=True)
    transformed_dataset = transformer.transform(dataset)
    expected_transformed_w = scaler.fit_transform(w)
    np.testing.assert_array_almost_equal(transformed_dataset.w, expected_transformed_w)

    # Test untransform on w
    with pytest.raises(NotImplementedError) as exception_info:
        untransformed_w = transformer.untransform(transformed_dataset.w)
    assert str(exception_info.value) == 'SklearnPipelineWrapper does not support inverse transforms'

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

def test_create_feature_transformers():
    """Test the `create_feature_transformers` when params.transformers is None."""

    params = pp.wrapper({})
    params.transformers = None
    transformers_x = trans.create_feature_transformers(
        params,
        featurization = None,
        train_dset = None
    )

    assert transformers_x == []

def test_zero_out_inf_nan_numpy_with_nonfinite_replaces_and_copies():
    x = np.array([1.0, np.nan, np.inf, -np.inf, 2.5], dtype=float)
    y = trans.zero_out_inf_nan(x)

    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, np.array([1.0, 0.0, 0.0, 0.0, 2.5], dtype=float))

    # Ensure it is a copy and original not mutated
    assert y is not x
    np.testing.assert_array_equal(x, np.array([1.0, np.nan, np.inf, -np.inf, 2.5], dtype=float))


def test_zero_out_inf_nan_numpy_without_nonfinite_no_change_but_copy():
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    y = trans.zero_out_inf_nan(x)

    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, x)
    assert y is not x


def test_zero_out_inf_nan_series_with_nonfinite_replaces_preserves_index_and_name():
    s = pd.Series([1.0, np.nan, np.inf, -np.inf], index=["a", "b", "c", "d"], name="vals")
    out = trans.zero_out_inf_nan(s)

    assert isinstance(out, pd.Series)
    assert out.name == "vals"
    assert list(out.index) == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(out.values, np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    # Original not mutated
    assert np.isnan(s.loc["b"])
    assert np.isposinf(s.loc["c"])
    assert np.isneginf(s.loc["d"])


def test_zero_out_inf_nan_series_without_nonfinite_no_change_values():
    s = pd.Series([1.0, 2.0], index=[10, 20], name="ok")
    out = trans.zero_out_inf_nan(s)

    assert isinstance(out, pd.Series)
    assert out.name == "ok"
    assert list(out.index) == [10, 20]
    np.testing.assert_array_equal(out.values, s.values)


def test_zero_out_inf_nan_dataframe_with_nonfinite_replaces_preserves_index_and_columns():
    df = pd.DataFrame(
        {"c1": [1.0, np.nan], "c2": [np.inf, 4.0]},
        index=["r1", "r2"],
    )
    out = trans.zero_out_inf_nan(df)

    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["r1", "r2"]
    assert list(out.columns) == ["c1", "c2"]
    np.testing.assert_array_equal(out.values, np.array([[1.0, 0.0], [0.0, 4.0]], dtype=float))

    # Original not mutated
    assert np.isnan(df.loc["r2", "c1"])
    assert np.isposinf(df.loc["r1", "c2"])


def test_zero_out_inf_nan_dataframe_without_nonfinite_no_change_values():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=[0, 1])
    out = trans.zero_out_inf_nan(df)

    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == [0, 1]
    assert list(out.columns) == ["a", "b"]
    np.testing.assert_array_equal(out.values, df.values)