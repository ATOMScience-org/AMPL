import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.predict_from_model as pfm
import os
import shutil
import glob
import pandas as pd

def test_multiclass_prediction():
    result_dir = os.path.dirname(__file__)
    result_dir = os.path.join(result_dir, 'result')

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    example_file = "./example.csv"
    id_col = "Id"
    smiles_col = "smiles"
    class_number = 3

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
            "max_epochs": "50",
            
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

    # Find resulting tarball
    files = glob.glob(os.path.join(result_dir, '*.tar.gz'))
    assert len(files) == 1

    tar_file = files[0]
    example_df = pd.read_csv(example_file)

    preds = pfm.predict_from_model_file(tar_file, example_df, id_col=id_col, smiles_col=smiles_col)
    print(preds.shape)
    print('DONE!!!')

if __name__ == '__main__':
    test_multiclass_prediction()