import tempfile

import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp
import numpy as np

import logging
logger = logging.getLogger(__name__)

def test_balancing_transformer():
    dset_key = '../../test_datasets/MRP3_dataset.csv'

    res_dir = tempfile.mkdtemp()

    balanced_params = params_w_balan(dset_key, res_dir)
    balanced_weights = make_pipeline_and_get_weights(balanced_params)
    (major_weight, minor_weight), (major_count, minor_count) = np.unique(balanced_weights, return_counts=True)
    assert major_weight < minor_weight
    assert major_count > minor_count

    nonbalanced_params = params_wo_balan(dset_key, res_dir)
    nonbalanced_weights = make_pipeline_and_get_weights(nonbalanced_params)
    (weight,), (count,) = np.unique(nonbalanced_weights, return_counts=True)
    assert weight == 1
    assert count == 436

def make_pipeline_and_get_weights(params):
    pparams = parse.wrapper(params)
    model_pipeline = mp.ModelPipeline(pparams)
    model_pipeline.train_model()

    return model_pipeline.data.train_valid_dsets[0][0].w

def params_wo_balan(dset_key, res_dir):
    # Train classification models without balancing weights. Repeat this several times so we can get some statistics on the performance metrics.
    params = {
        "dataset_key" : dset_key,
        "datastore" : "False",
        "uncertainty": "False",
        "splitter": "scaffold",
        "split_valid_frac": "0.20",
        "split_test_frac": "0.20",
        "split_strategy": "train_valid_test",
        "prediction_type": "classification",
        "model_choice_score_type": "roc_auc",
        "response_cols" : "active",
        "id_col": "compound_id",
        "smiles_col" : "rdkit_smiles",
        "result_dir": res_dir,
        "system": "LC",
        "transformers": "True",
        "model_type": "NN",
        "featurizer": "ecfp",
        "learning_rate": ".0007",
        "layer_sizes": "512,128",
        "dropouts": "0.3,0.3",
        "save_results": "False",
        "max_epochs": "2", # You don't need to train very long. Just need to build datasets
        "early_stopping_patience": "2",
        "verbose": "False",
        "seed":"0",
    }

    return params

def params_w_balan(dset_key, res_dir):
    # Now train models on the same dataset with balancing weights
    params = {
        "dataset_key" : dset_key,
        "datastore" : "False",
        "uncertainty": "False",
        "splitter": "scaffold",
        "split_valid_frac": "0.20",
        "split_test_frac": "0.20",
        "split_strategy": "train_valid_test",
        "prediction_type": "classification",
        "model_choice_score_type": "roc_auc",
        "response_cols" : "active",
        "id_col": "compound_id",
        "smiles_col" : "rdkit_smiles",
        "result_dir": res_dir,
        "system": "LC",
        "transformers": "True",
        "model_type": "NN",
        "featurizer": "computed_descriptors",
        "descriptor_type": "rdkit_raw",
        "weight_transform_type": "balancing",
        "learning_rate": ".0007",
        "layer_sizes": "512,128",
        "dropouts": "0.3,0.3",
        "save_results": "False",
        "max_epochs": "2",
        "early_stopping_patience": "2",
        "verbose": "False",
        "seed":"0",
    }

    return params

if __name__ == '__main__':
    test_balancing_transformer()