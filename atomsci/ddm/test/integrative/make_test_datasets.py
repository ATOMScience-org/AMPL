import os
import numpy as np
import pandas as pd

def get_absolute_path(relative_path):
    """
    Calculate the absolute path given a path relative to the location of this file.

    Parameters:
    relative_path (str): The relative path to convert.

    Returns:
    str: The absolute path.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Join the current directory with the relative path
    absolute_path = os.path.join(current_dir, relative_path)
    return absolute_path

def get_features(feature_type):
    desc_spec_file = get_absolute_path('../../data/descriptor_sets_sources_by_descr_type.csv')
    desc_spec_df = pd.read_csv(desc_spec_file, index_col=False)

    rows = desc_spec_df[desc_spec_df['descr_type']==feature_type]
    assert len(rows)==1
    descriptors = rows['descriptors'].values[0].split(';')

    return descriptors

def rescale(array, desired_mean, desired_std):
    """
    Rescale the input array to have the desired mean and standard deviation.

    Parameters:
    array (np.ndarray): Input array to be rescaled.
    desired_mean (float): Desired mean of the rescaled array.
    desired_std (float): Desired standard deviation of the rescaled array.

    Returns:
    np.ndarray: Rescaled array with the desired mean and standard deviation.
    """
    current_mean = np.mean(array)
    current_std = np.std(array)
    
    # Rescale the array
    rescaled_array = (array - current_mean) / current_std * desired_std + desired_mean
    
    return rescaled_array

def make_test_dataset(features, num_train=50000, num_test=10000, num_valid=10000):
    """
    Create a test dataset with specified features and splits.

    Parameters:
    features (list): List of column names for the features.
    num_train (int): Number of training samples.
    num_test (int): Number of test samples.
    num_valid (int): Number of validation samples.

    Returns:
    pd.DataFrame: DataFrame with the specified features and an additional 'id' column.
    np.ndarray: IDs for the training split.
    np.ndarray: IDs for the test split.
    np.ndarray: IDs for the validation split.
    """
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Generate IDs
    total_samples = num_train + num_test + num_valid
    compounds_ids = np.arange(total_samples)

    # Populate feature columns with numbers from different normal distributions
    dat = {'compound_id':compounds_ids}
    for feature in features:
        train_data = np.random.normal(loc=0, scale=1, size=num_train)
        train_data = rescale(train_data, 0, 2)
        valid_data = np.random.normal(loc=0, scale=1, size=num_valid)
        valid_data = rescale(valid_data, 10, 5)
        test_data = np.random.normal(loc=0, scale=1, size=num_test)
        test_data = rescale(test_data, 100, 10)
        dat[feature] = np.concatenate([train_data, test_data, valid_data])

    # Add response columns with different means
    dat['response_1'] = np.random.normal(loc=0, scale=1, size=total_samples)
    dat['response_2'] = np.random.normal(loc=100, scale=1, size=total_samples)

    # Add class column
    class_labels = np.zeros(total_samples)
    dat['class'] = class_labels

    df = pd.DataFrame(data=dat)

    # Ensure each subset has the same ratio of 0s and 1s in the class column
    train_class = np.concatenate([
        np.zeros(int(num_train * 0.8)),
        np.ones(int(num_train * 0.2))
    ])
    np.random.shuffle(train_class)
    df.loc[:num_train-1, 'class'] = train_class

    valid_class = np.concatenate([
        np.zeros(int(num_valid * 0.5)),
        np.ones(int(num_valid * 0.5))
    ])
    np.random.shuffle(valid_class)
    df.loc[num_train:num_train+num_valid-1, 'class'] = valid_class

    test_class = np.concatenate([
        np.zeros(int(num_test * 0.5)),
        np.ones(int(num_test * 0.5))
    ])
    np.random.shuffle(test_class)
    df.loc[num_train+num_valid:, 'class'] = test_class

    # Split IDs
    train_ids = df['compound_id'][:num_train].values
    test_ids = df['compound_id'][num_train:num_train + num_test].values
    valid_ids = df['compound_id'][num_train + num_test:].values

    return df, train_ids, valid_ids, test_ids

def make_split_df(train_ids, valid_ids, test_ids):
    """
    Create a DataFrame with train, valid, and test IDs, and additional columns for subset and fold.

    Parameters:
    train_ids (list or np.ndarray): List of training IDs.
    valid_ids (list or np.ndarray): List of validation IDs.
    test_ids (list or np.ndarray): List of test IDs.

    Returns:
    pd.DataFrame: DataFrame with 'id', 'subset', and 'fold' columns.
    """
    # Create DataFrames for each subset
    train_df = pd.DataFrame({'cmpd_id': train_ids, 'subset': 'train', 'fold': 0})
    valid_df = pd.DataFrame({'cmpd_id': valid_ids, 'subset': 'valid', 'fold': 0})
    test_df = pd.DataFrame({'cmpd_id': test_ids, 'subset': 'test', 'fold': 0})

    # Concatenate the DataFrames
    split_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    return split_df

def make_test_dataset_and_split(dataset_key, feature_types):
    features = get_features(feature_types)
    df, train_ids, valid_ids, test_ids = make_test_dataset(features)

    df['rdkit_smiles'] = ['CCCC'] * len(df)
    split_df = make_split_df(train_ids, valid_ids, test_ids)

    df.to_csv(dataset_key, index=False)

    split_filename = os.path.splitext(dataset_key)[0]+'_train_valid_test_index_testsplit.csv'
    split_df.to_csv(split_filename, index=False)