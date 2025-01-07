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
    """
    Gets the feature columns given a feature_type, e.g. rdkit_raw
    """
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

def make_test_dataset(features, num_train=500, num_test=100, num_valid=100):
    """
    Create a test dataset with specified features and splits.

    Parameters:
    features (list): List of column names for the features.
    num_train (int): Number of training samples.
    num_test (int): Number of test samples.
    num_valid (int): Number of validation samples.

    Returns:
    pd.DataFrame: DataFrame with the specified features and additional columns.
    np.ndarray: IDs for the training split.
    np.ndarray: IDs for the test split.
    np.ndarray: IDs for the validation split.
    """
    # Generate data for each subset
    train_df = generate_subset_df(num_train, features, feature_mean=0, feature_std=2, 
                                  response1_mean=0, response1_std=1,
                                  response2_mean=100, response2_std=1,
                                  positive_frac=0.2)
    valid_df = generate_subset_df(num_valid, features, feature_mean=10, feature_std=5, 
                                  response1_mean=0, response1_std=1, 
                                  response2_mean=100, response2_std=1, 
                                  positive_frac=0.5)
    test_df = generate_subset_df(num_test, features, feature_mean=100, feature_std=10, 
                                  response1_mean=0, response1_std=1, 
                                  response2_mean=100, response2_std=1, 
                                  positive_frac=0.5)

    # Concatenate the dataframes
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    df['compound_id'] = list(range(len(df)))

    # Split IDs
    train_ids = df['compound_id'][:num_train].values
    valid_ids = df['compound_id'][num_train:num_train + num_valid].values
    test_ids = df['compound_id'][num_train + num_valid:].values

    return df, train_ids, valid_ids, test_ids

def generate_subset_df(fold_size, features, 
                       feature_mean, feature_std, 
                       response1_mean=0, response1_std=1, 
                       response2_mean=100, response2_std=10,
                       positive_frac=0.8):
    """
    Generate a fold of data with specified parameters.

    Parameters:
    fold_size (int): Number of samples in the fold.
    features (list): List of feature names.
    feature_mean (float): Mean of the feature data.
    feature_std (float): Standard deviation of the feature data.
    response1_mean (float): Mean of the response_1 data.
    response1_std (float): Standard deviation of the response_1 data.
    response2_mean (float): Mean of the response_2 data.
    response2_std (float): Standard deviation of the response_2 data.
    positive_frac (float): Fraction of positive class labels.

    Returns:
    pd.DataFrame: DataFrame containing the generated fold data.
    """
    dat = {}
    for feature in features:
        fold_data = np.random.normal(loc=0, scale=1, size=fold_size)
        fold_data = rescale(fold_data, feature_mean, feature_std)
        dat[feature] = fold_data

    # Add response columns with different means
    response_1 = np.random.normal(loc=0, scale=1, size=fold_size)
    dat['response_1'] = rescale(response_1, response1_mean, response1_std)
    response_2 = np.random.normal(loc=response2_mean, scale=response2_std, size=fold_size)
    dat['response_2'] = rescale(response_2, response2_mean, response2_std)

    # Add class column
    # Ensure each subset has the same ratio of 0s and 1s in the class column
    class_labels = np.concatenate([
        np.zeros(int(fold_size * (1 - positive_frac))),
        np.ones(int(fold_size * positive_frac))
    ])
    np.random.shuffle(class_labels)
    dat['class'] = class_labels

    return pd.DataFrame(data=dat)

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
    """
    Given a dataset key and a feature type, create a featurized
    csv file and split.

    Args:
        dataset_key (str): Where to save the newly generated test DataFrame

        feature_types (str): A feature type, e.g. rdkit_raw


    """
    features = get_features(feature_types)
    df, train_ids, valid_ids, test_ids = make_test_dataset(features)

    df['rdkit_smiles'] = ['CCCC'] * len(df)
    split_df = make_split_df(train_ids, valid_ids, test_ids)

    df.to_csv(dataset_key, index=False)

    split_filename = os.path.splitext(dataset_key)[0]+'_train_valid_test_index_testsplit.csv'
    split_df.to_csv(split_filename, index=False)

def make_kfold_test_dataset(features, fold_size=1000, num_test=1000, num_folds=5):
    """
    Create a k-fold test dataset with specified features and splits.

    Parameters:
    features (list): List of column names for the features.
    fold_size (int): Number of samples in each fold.
    num_test (int): Number of test samples.
    num_folds (int): Number of folds.

    Returns:
    pd.DataFrame: DataFrame with the specified features and additional columns.
    list: List of lists containing IDs for each fold.
    np.ndarray: IDs for the test split.
    """
    # Populate feature columns with numbers from different normal distributions
    fold_dfs = []
    for f in range(num_folds):
        fold_dfs.append(generate_subset_df(fold_size, features,
                           feature_mean=f*f, feature_std=f+1,
                           response1_mean=f*f, response1_std=f+1,
                           response2_mean=f*f*10, response2_std=(f+1)*10,
                           positive_frac=0.2))

    # generate test_df
    fold_dfs.append(generate_subset_df(num_test, features,
            feature_mean=num_folds*num_folds, feature_std=num_folds+2,
            response1_mean=num_folds*num_folds, response1_std=num_folds+2,
            response2_mean=(num_folds*num_folds)*10, response2_std=(num_folds+2)*10,
            positive_frac=0.5))

    df = pd.concat(fold_dfs)
    df['compound_id'] = list(range(len(df)))

    train_valid_ids = []
    for f in range(num_folds):
        train_valid_ids.append(list(range(f*fold_size, (f+1)*fold_size)))

    test_ids = df['compound_id'].values[-num_test:]

    return df, train_valid_ids, test_ids

def make_kfold_split_df(train_valid_ids, test_ids):
    """
    Given lists of train_valid_ids and test_ids create a split DataFrame

    Args:
        train_valid_ids (list of lists): A list of ids

        test_ids (list): Ids for test compounds

    Returns:
        DataFrame: A split DataFrame that can be read by AMPL
    """
    fold_dfs = []
    for i, tvi in enumerate(train_valid_ids):
        data = {'cmpd_id':tvi}
        data['subset'] = ['train_valid']*len(tvi)
        data['fold'] = [i]*len(tvi)

        fold_dfs.append(pd.DataFrame(data=data))

    data = {'cmpd_id':test_ids}
    data['subset'] = ['test']*len(test_ids)
    data['fold'] = [0]*len(test_ids)
    
    fold_dfs.append(pd.DataFrame(data=data))

    split_df = pd.concat(fold_dfs)

    return split_df

def make_kfold_dataset_and_split(dataset_key, feature_types, num_folds=3):
    """
    Given a dataset key and a feature type,  and nubmer of folds create a featurized
    csv file and split.

    Args:
        dataset_key (str): Where to save the newly generated test DataFrame

        feature_types (str): A feature type, e.g. rdkit_raw

        num_folds (int): Number of folds

    """
    features = get_features(feature_types)
    df, train_ids, test_ids = make_kfold_test_dataset(features, num_folds=num_folds)

    df['rdkit_smiles'] = ['CCCC'] * len(df)
    split_df = make_kfold_split_df(train_ids, test_ids)

    df.to_csv(dataset_key, index=False)

    split_filename = os.path.splitext(dataset_key)[0]+f'_{num_folds}_fold_cv_index_testsplit.csv'
    split_df.to_csv(split_filename, index=False)

