import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.compare_models as cm
import atomsci.ddm.utils.test_utils as tu
import atomsci.ddm.utils.model_retrain as mr
import os
import shutil
import pandas as pd

def clean(result_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

def test_production():
    """Train a model in production mode"""
    result_dir = tu.relative_to_file(__file__, 'result')
    clean(result_dir)

    id_col = "Id"
    smiles_col = "smiles"
    class_number = 3
    example_file = tu.relative_to_file(__file__, './example.csv')
    max_epochs = 50

    config_json = \
        {
            "comment": "System",
            "comment": "----------------------------------------",
            "system": "LC",
            "datastore": "False",
            "save_results": "False",
            "data_owner": "username",

            "comment": "Input file",
            "comment": "----------------------------------------",
            "comment": "Note: dataset_key must be a path/file name: E.G. ./dataset.csv",
            "dataset_key": example_file,
            "id_col": id_col,
            "smiles_col": smiles_col,
            "class_number": class_number,
            "production": "True",

            "comment": "Split",
            "comment": "----------------------------------------",
            "splitter": "random",
            "previously_split": "True",
            "split_uuid": "bad-split",

            "comment": "Prediction Type",
            "comment": "----------------------------------------",
            "response_cols": "sol_category",
            "prediction_type": "classification",

            "comment": "Features",
            "comment": "----------------------------------------",
            "featurizer": "ecfp",

            "comment": "Model",
            "comment": "----------------------------------------",
            "model_type": "NN",
            "dropout": ".01,.01,.01",
            "layer_sizes": "256,50,18",
            "learning_rate": "0.00007",
            "max_epochs": f"{max_epochs}",
            
            "comment": "Training",
            "comment": "----------------------------------------",
            "comment": "This regulates how long to train the model",
            "early_stopping_patience": "2",

            "comment": "Results",
            "comment": "----------------------------------------",    
            "result_dir": result_dir
        }

    # Parse parameters
    params = parse.wrapper(config_json)

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    result_df = cm.get_filesystem_perf_results(result_dir)

    # we cleaned the result dir before running, 
    # so this should just have one result
    assert len(result_df) == 1

    # train size should equal valid size should equal test size
    # should equal size of the dataset
    expected_size = len(pd.read_csv(example_file))
    assert result_df.best_valid_num_compounds[0] == expected_size
    assert result_df.best_test_num_compounds[0] == expected_size
    assert result_df.best_train_num_compounds[0] == expected_size

    # best_epoch should equal max_epoch-1, since we start at 0
    assert result_df.best_epoch[0] == (max_epochs-1)

def test_production_retrain():
    """retrain a model in production mode"""
    result_dir = tu.relative_to_file(__file__, 'result')
    clean(result_dir)
    new_result_dir = tu.relative_to_file(__file__, 'new_result')
    clean(new_result_dir)

    id_col = "Id"
    smiles_col = "smiles"
    class_number = 3
    example_file = tu.relative_to_file(__file__, './example.csv')
    max_epochs = 50

    config_json = \
        {
            "comment": "System",
            "comment": "----------------------------------------",
            "system": "LC",
            "datastore": "False",
            "save_results": "False",
            "data_owner": "username",

            "comment": "Input file",
            "comment": "----------------------------------------",
            "comment": "Note: dataset_key must be a path/file name: E.G. ./dataset.csv",
            "dataset_key": example_file,
            "id_col": id_col,
            "smiles_col": smiles_col,
            "class_number": class_number,

            "comment": "Split",
            "comment": "----------------------------------------",
            "splitter": "random",

            "comment": "Prediction Type",
            "comment": "----------------------------------------",
            "response_cols": "sol_category",
            "prediction_type": "classification",

            "comment": "Features",
            "comment": "----------------------------------------",
            "featurizer": "ecfp",

            "comment": "Model",
            "comment": "----------------------------------------",
            "model_type": "NN",
            "dropout": ".01,.01,.01",
            "layer_sizes": "256,50,18",
            "learning_rate": "0.00007",
            "max_epochs": f"{max_epochs}",
            
            "comment": "Training",
            "comment": "----------------------------------------",
            "comment": "This regulates how long to train the model",
            "early_stopping_patience": "20",

            "comment": "Results",
            "comment": "----------------------------------------",    
            "result_dir": result_dir
        }

    # Parse parameters
    params = parse.wrapper(config_json)

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    result_df = cm.get_filesystem_perf_results(result_dir)

    # we cleaned the result dir before running, 
    # so this should just have one result
    assert len(result_df) == 1

    best_tar = result_df.model_path[0]
    best_epoch = result_df.best_epoch[0]
    mr.train_model_from_tar(best_tar, new_result_dir, production=True)

    new_result_df = cm.get_filesystem_perf_results(new_result_dir)

    # we cleaned the result dir before running, 
    # so this should just have one result
    assert len(new_result_df) == 1

    # train size should equal valid size should equal test size
    # should equal size of the dataset
    expected_size = len(pd.read_csv(example_file))
    assert new_result_df.best_valid_num_compounds[0] == expected_size
    assert new_result_df.best_test_num_compounds[0] == expected_size
    assert new_result_df.best_train_num_compounds[0] == expected_size

    # best_epoch should equal max_epoch-1, since we start at 0
    assert new_result_df.best_epoch[0] == best_epoch

if __name__ == '__main__':
    test_production_retrain()