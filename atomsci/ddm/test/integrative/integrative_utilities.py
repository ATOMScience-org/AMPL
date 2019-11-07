import glob
import json
import os
import requests
import shutil


def clean_fit_predict():
    """
    Clean integrative test files
    """

    split_files = glob.glob('./*_train_valid_test_*')
    for f in split_files:
        os.remove(f)

    if os.path.exists('result'):
        shutil.rmtree('result')

    if os.path.exists('predict'):
        shutil.rmtree('predict')


def download_save(url, file_name, verify=True):
    """
    Download dataset

    Arguments:
        url: URL
        file_name: File name
        verify: Verify SSL certificate
    """

    # Skip if file already exists
    if (not (os.path.isfile(file_name) and os.path.getsize(file_name) > 0)):
        assert(url)
        r = requests.get(url, verify=verify)

        assert (r.status_code == 200), 'Error: Could not download dataset'

        with open(file_name, 'wb') as f:
            f.write(r.content)

        assert (os.path.isfile(file_name) and os.path.getsize(file_name) > 0), 'Error: dataset file not created'


def get_subdirectory(model_dir_root):
    """
    Get model uuid from model directory with uuid subdirectory. Assumes that there is only one subdirectory.
    model_dir_root: Model directory with uuid subdirectory

    Returns:
         uuid: Model uuid
    """

    # Get subdirectory: model uuid
    walker = os.walk(model_dir_root)
    dirs = [d for d in walker]
    uuid = dirs[0][1][0]

    return uuid


def training_statistics_file(model_dir, subset, minimum_r2):
    """
    Get training statistics

    Arguments:
        model_dir: Model directory with training_model_metrics.json
        subset: Data subset
        minimum_r2: Minimum R^2
    """

    # Open training JSON file
    assert (os.path.exists(model_dir)), 'Error: Result directory does not exist'

    # Open training model metrics file
    training_model_metrics_file = model_dir+'/training_model_metrics.json'
    assert (os.path.exists(training_model_metrics_file)), 'Error: Model metadata file does not exist'
    with open(training_model_metrics_file) as f:
        training_model_metrics = json.loads(f.read())

    # Get best statistics
    training_run = training_model_metrics['ModelMetrics']['TrainingRun']
    for t in training_run:
        if (t['subset'] == subset) and (t['label'] == 'best'):
            break

    assert (t['PredictionResults']['r2_score'] >= minimum_r2), 'Error: Model test R^2 < minimum R^2'
