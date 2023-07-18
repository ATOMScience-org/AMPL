import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.predict_from_model as pfm
import atomsci.ddm.utils.test_utils as tu
import os
import shutil
import glob
import pandas as pd

def clean(result_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

def test_production():
    '''
    Every train, valid, test, split needs to have all classes.
    In this test the test subset does not have any 1 class samples
    '''
    result_dir = tu.relative_to_file(__file__, 'result')
    clean(result_dir)

    id_col = "Id"
    smiles_col = "smiles"
    class_number = 3
    example_file = tu.relative_to_file(__file__, './example.csv')

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
            "max_epochs": "10",
            
            "comment": "Training",
            "comment": "----------------------------------------",
            "comment": "This regulates how long to train the model",
            "early_stopping_patience": "50",

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

if __name__ == '__main__':
    test_production()