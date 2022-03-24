import pandas as pd
import tempfile

import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.model_pipeline as mp

import logging
logger = logging.getLogger(__name__)

nreps = 10
metrics = []
vals = []
balanced = []
subset = []

def test_balancing_transformer():
    dset_key = '../../test_datasets/MRP3_dataset.csv'
    dset_df = pd.read_csv(dset_key)

    res_dir = tempfile.mkdtemp()
    split_uuid = create_scaffold_split(dset_key, res_dir)

    # train the model without the balancing
    train_model_wo_balan(dset_key, split_uuid, res_dir)
    # train the model with the balancing parameter
    train_model_w_balan(dset_key, split_uuid, res_dir)

    metrics_df = pd.DataFrame(dict(subset=subset, balanced=balanced, metric=metrics, val=vals))

    # check the recall_score
    rec_df = metrics_df[metrics_df.metric == 'recall_score']

    not_balanced_series = rec_df[(rec_df.balanced == 'no')].groupby("subset").val.mean()
    balanced_series = rec_df[(rec_df.balanced == 'yes')].groupby("subset").val.mean()

    assert((balanced_series['test'] > not_balanced_series['test']) & (balanced_series['valid'] > not_balanced_series['valid']) )

def create_scaffold_split(dset_key, res_dir):
    params = {
        "dataset_key" : dset_key,
        "datastore" : "False",
        "uncertainty": "False",
        "splitter": "scaffold",
        "split_valid_frac": "0.1",
        "split_test_frac": "0.1",
        "split_strategy": "train_valid_test",
        "previously_split": "False",
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
        "learning_rate": ".0007",
        "layer_sizes": "512,128",
        "dropouts": "0.3,0.3",
        "save_results": "False",
        "max_epochs": "500",
        "early_stopping_patience": "50",
        "verbose": "False"
    }

    pparams = parse.wrapper(params)
    MP = mp.ModelPipeline(pparams)

    split_uuid = MP.split_dataset()
    return split_uuid

def train_model_wo_balan(dset_key, split_uuid, res_dir):
    # Train classification models without balancing weights. Repeat this several times so we can get some statistics on the performance metrics.
    params = {
        "dataset_key" : dset_key,
        "datastore" : "False",
        "uncertainty": "False",
        "splitter": "scaffold",
        "split_valid_frac": "0.1",
        "split_test_frac": "0.1",
        "split_strategy": "train_valid_test",
        "previously_split": "True",
        "split_uuid": split_uuid,
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
        "learning_rate": ".0007",
        "layer_sizes": "512,128",
        "dropouts": "0.3,0.3",
        "save_results": "False",
        "max_epochs": "500",
        "early_stopping_patience": "50",
        "verbose": "False"
    }

    for i in range(nreps):
        pparams = parse.wrapper(params)
        MP = mp.ModelPipeline(pparams)
        MP.train_model()
        wrapper = MP.model_wrapper

        for ss in ['valid', 'test']:
            metvals = wrapper.get_pred_results(ss, 'best')
            for metric in ['roc_auc_score', 'prc_auc_score', 'cross_entropy', 'precision', 'recall_score', 'npv', 'accuracy_score', 'bal_accuracy', 'kappa','matthews_cc']:
                subset.append(ss)
                balanced.append('no')
                metrics.append(metric)
                vals.append(metvals[metric])

def train_model_w_balan(dset_key, split_uuid, res_dir):
    # Now train models on the same dataset with balancing weights
    params = {
        "dataset_key" : dset_key,
        "datastore" : "False",
        "uncertainty": "False",
        "splitter": "scaffold",
        "split_valid_frac": "0.1",
        "split_test_frac": "0.1",
        "split_strategy": "train_valid_test",
        "previously_split": "True",
        "split_uuid": split_uuid,
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
        "max_epochs": "500",
        "early_stopping_patience": "50",
        "verbose": "False"
    }

    for i in range(nreps):
        pparams = parse.wrapper(params)
        MP = mp.ModelPipeline(pparams)
        MP.train_model()
        wrapper = MP.model_wrapper

        for ss in ['valid', 'test']:
            metvals = wrapper.get_pred_results(ss, 'best')
            for metric in ['roc_auc_score', 'prc_auc_score', 'cross_entropy', 'precision', 'recall_score', 'npv', 'accuracy_score', 'bal_accuracy', 'kappa','matthews_cc']:
                subset.append(ss)
                balanced.append('yes')
                metrics.append(metric)
                vals.append(metvals[metric])

if __name__ == '__main__':
    test_balancing_transformer()