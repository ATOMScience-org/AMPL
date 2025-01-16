# This tests using an existing NN model to 
# generate embedding features.
import os
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import shutil

def test_embedded_features():

    dskey = os.path.join(os.path.dirname(__file__), 
                         '../../test_datasets/H1_hybrid.csv')
    id_col = 'compound_id'
    smiles_col = 'rdkit_smiles'
    response_cols = 'activity'

    model_path = os.path.join(os.path.dirname(__file__), 
        'embedding_model.tar.gz')
    result_dir = os.path.join(os.path.dirname(__file__),
        'test_embedding')

    transfer_json = {
        "dataset_key" : dskey,
        "datastore" : "False",
        "uncertainty": "False",
        "splitter": 'scaffold',
        "split_valid_frac": "0.15",
        "split_test_frac": "0.15",
        "split_strategy": "train_valid_test",
        "prediction_type": "regression",
        "model_choice_score_type": "r2",
        "response_cols" : response_cols,
        "id_col": id_col,
        "smiles_col" : smiles_col,
        "result_dir": result_dir,
        "system": "LC",
        "transformers": "True",
        "model_type": "NN",
        "featurizer": "embedding",
        "embedding_model_path": model_path,
        "learning_rate": .0007,
        "layer_sizes": [128, 512, 128, 64],
        "dropouts": [0.2, 0.2, 0.2, 0.2],
        "max_epochs": "5",
        "early_stopping_patience": "5",
        "verbose": "False"
    }

    pparams = parse.wrapper(transfer_json)

    pparams = parse.wrapper(transfer_json)
    model_pipeline = mp.ModelPipeline(pparams)
    model_pipeline.train_model()

    # clean
    shutil.rmtree(result_dir)
    print('cleaned and done')

if __name__ == '__main__':
    test_embedded_features()