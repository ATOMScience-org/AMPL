import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.predict_from_model as pfm
from atomsci.ddm.pipeline.model_datasets import ClassificationDataException
import atomsci.ddm.utils.test_utils as tu
import os
import shutil
import glob
import pandas as pd

def clean(result_dir):
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

def _example_json(example_file, result_dir,
                id_col, smiles_col, class_number):

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
            "max_epochs": "10",
            
            "comment": "Training",
            "comment": "----------------------------------------",
            "comment": "This regulates how long to train the model",
            "early_stopping_patience": "50",

            "comment": "Results",
            "comment": "----------------------------------------",    
            "result_dir": result_dir
        }

    return config_json

def test_multiclass_prediction():
    example_file = tu.relative_to_file(__file__, 'example.csv')
    result_dir = tu.relative_to_file(__file__, 'result')
    clean(result_dir)

    id_col = "Id"
    smiles_col = "smiles"
    class_number = 3

    config_json = _example_json(example_file, result_dir,
                                id_col, smiles_col, class_number)

    # Parse parameters
    params = parse.wrapper(config_json)

    # Create model pipeline
    model = mp.ModelPipeline(params)

    # Train model
    model.train_model()

    # Find resulting tarball
    files = glob.glob(os.path.join(result_dir, '*.tar.gz'))
    assert len(files) == 1

    tar_file = files[0]
    example_df = pd.read_csv(example_file)

    preds = pfm.predict_from_model_file(tar_file, example_df, id_col=id_col, smiles_col=smiles_col)
    print(preds.shape)
    print(preds.columns)
    preds.to_csv(tu.relative_to_file(__file__, 'preds.csv'))
    print('DONE!!!')

def test_out_of_range():
    result_dir = tu.relative_to_file(__file__, 'result')
    clean(result_dir)

    id_col = "Id"
    smiles_col = "smiles"
    class_number = 3

    config_json = _example_json(
        tu.relative_to_file(__file__, "./example_out_of_range.csv"), 
        result_dir,
        id_col, smiles_col, class_number)

    # Parse parameters
    params = parse.wrapper(config_json)

    try:
        # Create model pipeline
        model = mp.ModelPipeline(params)

        # Train model
        model.train_model()
    except ClassificationDataException as err:
        assert "does not have all classes labeled using positive integers 0 <= i" in err.args[0]

    print('DONE!!!')

def test_bad_split():
    """Every train, valid, test, split needs to have all classes.
    In this test the test subset does not have any 1 class samples
    """
    result_dir = tu.relative_to_file(__file__, 'result')
    clean(result_dir)

    id_col = "Id"
    smiles_col = "smiles"
    class_number = 3
    example_file = tu.relative_to_file(__file__, './example_bad_split.csv')

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

    try:
        # Create model pipeline
        model = mp.ModelPipeline(params)

        # Train model
        model.train_model()
    except ClassificationDataException as err:
        assert "does not have all classes represented in a split" in err.args[0]

if __name__ == '__main__':
    test_bad_split()
    test_out_of_range()
    test_multiclass_prediction()