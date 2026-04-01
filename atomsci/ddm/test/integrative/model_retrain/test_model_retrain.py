import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.compare_models as cm
import atomsci.ddm.utils.model_file_reader as mfr
import atomsci.ddm.utils.test_utils as tu
import atomsci.ddm.utils.model_retrain as mr
import os
import shutil
import glob
import json

def clean(result_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

def train_model(result_dir):
    """Train a model in production mode"""

    json_file = tu.relative_to_file(__file__, './config.json')
    example_file = tu.relative_to_file(__file__, './example.csv')

    with open(json_file, 'r') as f:
        config_json = json.load(f)
        config_json['dataset_key'] = example_file
        config_json['result_dir'] = result_dir

    # Parse parameters
    params = parse.wrapper(config_json)

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

def retrain_model(model_tar, new_result_dir, keep_seed):
    """Retrains a model"""
    mr.train_model_from_tar(model_tar, new_result_dir, keep_seed=keep_seed)

def run_test_retrain(keep_seed):
    """Trains and retrains a model
    
    Trains and retrains a model and compares the results
    """

    # train a model
    result_dir = tu.relative_to_file(__file__, 'result')
    train_model(result_dir)

    # find the tar file
    result_df = cm.get_filesystem_perf_results(result_dir)
    assert(len(result_df) == 1)
    model_tar = result_df['model_path'].values[0]

    # retrain the model
    new_result_dir = tu.relative_to_file(__file__, 'retrain_result')
    retrain_model(model_tar, new_result_dir, keep_seed)

    # find the new tar file
    result_df = cm.get_filesystem_perf_results(new_result_dir)
    assert(len(result_df) == 1)
    new_model_tar = result_df['model_path'].values[0]

    original_model = mfr.ModelFileReader(model_tar)
    new_model = mfr.ModelFileReader(new_model_tar)

    assert new_model.get_split_uuid() == original_model.get_split_uuid()

    if keep_seed:
        assert new_model.get_random_seed()==original_model.get_random_seed()
    else:
        assert new_model.get_random_seed()!=original_model.get_random_seed()

    # clean files
    split_files = glob.glob(tu.relative_to_file(__file__, './example_*_random_*.csv'))
    for sf in split_files:
        os.remove(sf)
    clean(new_result_dir)
    clean(result_dir)

def test_retrain():
    run_test_retrain(True)
    run_test_retrain(False)

if __name__ == '__main__':
    test_retrain()