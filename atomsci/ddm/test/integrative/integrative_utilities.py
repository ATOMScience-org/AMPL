import glob
import json
import os
import shutil
import pandas as pd


def clean_fit_predict():
    """Clean integrative test files"""

    split_files = glob.glob('./*_train_valid_test_*')
    for f in split_files:
        os.remove(f)

    if os.path.exists('result'):
        shutil.rmtree('result')

    if os.path.exists('predict'):
        shutil.rmtree('predict')


def get_subdirectory(model_dir_root):
    """Get model uuid from model directory with uuid subdirectory. Assumes that there is only one subdirectory.
    model_dir_root: Model directory with uuid subdirectory

    Returns:
         uuid: Model uuid
    """

    # Get subdirectory: model uuid
    walker = os.walk(model_dir_root)
    dirs = [d for d in walker]
    uuid = dirs[0][1][0]

    return uuid


def training_statistics_file(model_dir, subset, minimum_r2, metric_col='r2_score'):
    """Get training statistics

    Arguments:
        model_dir: Model directory with training_model_metrics.json
        subset: Data subset
        minimum_r2: Minimum R^2
        metric_col: Desired metric e.g. r2_score, accuracy_score
    """

    test_r2 = read_training_statistics_file(model_dir, subset, metric_col)
    assert (test_r2 >= minimum_r2), 'Error: Model test R^2 %0.3f < minimum R^2 %0.3f'%(test_r2, minimum_r2)

def read_training_statistics_file(model_dir, subset, metric_col):
    """Get training statistics

    Arguments:
        model_dir: Model directory with training_model_metrics.json
        subset: Data subset
        metric_col: Desired metric e.g. r2_score, accuracy_score
    """

    # Open training JSON file
    assert (os.path.exists(model_dir)), 'Error: Result directory does not exist'

    # Open training model metrics file
    training_model_metrics_file = model_dir+'/model_metrics.json'
    assert (os.path.exists(training_model_metrics_file)), 'Error: Model metadata file does not exist'
    with open(training_model_metrics_file) as f:
        training_model_metrics = json.loads(f.read())

    # Get best statistics
    for m in training_model_metrics:
        if (m['subset'] == subset) and (m['label'] == 'best'):
            break

    test_metric = m['prediction_results'][metric_col]
    return test_metric

def copy_delaney(dest='.'):
    """Copies the delaney-processed.csv file to destination

    Copies the delaney-processed.csv file to the given destination.
    """

    delaney_source = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../test_datasets/delaney-processed.csv'))

    shutil.copy(delaney_source, dest)

def extract_seed(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata.get('seed')

def modify_params_with_seed(pparams, seed):
    pparams.seed = seed
    return pparams

def get_test_set(dataset_key, split_csv, id_col):
    """
    Read the dataset key and split_uuid to split dataset into split components 
    
    Parameters: 
        - dataset_key: path to csv file of dataset
        - split_uuid: path to split csv file 
        - id_col: name of ID column
    
    Returns:
        - train, valid, test dataframe
    """
    df = pd.read_csv(dataset_key)
    split_df=pd.read_csv(split_csv)
    test_df = df[df[id_col].isin(split_df[split_df['subset']=='test']['cmpd_id'])]

    return test_df

def find_best_test_metric(model_metrics):
    for metric in model_metrics:
        if metric['label'] == 'best' and metric['subset']=='test':
            return metric 
    return None 
