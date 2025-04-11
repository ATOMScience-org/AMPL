# This tests using an existing NN model to 
# generate embedding features.
import os
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import shutil
import glob
import time

def test_embedded_features():

    dskey = os.path.join(os.path.dirname(__file__), 
                         '../../test_datasets/H1_hybrid.csv')
    expected_rdkit_raw_file = os.path.join(os.path.dirname(__file__), 
                         '../../test_datasets/scaled_descriptors/H1_hybrid_with_rdkit_raw_descriptors.csv')
    expected_mordred_filtered_file = os.path.join(os.path.dirname(__file__), 
                         '../../test_datasets/scaled_descriptors/H1_hybrid_with_mordred_filtered_descriptors.csv')

    if os.path.exists(expected_rdkit_raw_file):
        os.remove(expected_rdkit_raw_file)
    if os.path.exists(expected_mordred_filtered_file):
        os.remove(expected_mordred_filtered_file)

    assert not os.path.exists(expected_rdkit_raw_file), f'{expected_rdkit_raw_file} should not exist'
    assert not os.path.exists(expected_mordred_filtered_file), f'{expected_mordred_filtered_file} should not exist'

    id_col = 'compound_id'
    smiles_col = 'rdkit_smiles'
    response_cols = 'activity'

    model_paths = glob.glob( os.path.join(os.path.dirname(__file__), 
        'embedding_model*.tar.gz'))
    result_dir = os.path.join(os.path.dirname(__file__),
        'test_embedding')

    def run_tests():
        split_uuids = []
        for model_path in model_paths:
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

            model_pipeline = mp.ModelPipeline(pparams)
            model_pipeline.train_model()
            split_uuids.append(model_pipeline.data.split_uuid)

        return split_uuids

    start = time.time()
    all_split_uuids = run_tests()
    end = time.time()
    first_run = end-start

    assert os.path.exists(expected_rdkit_raw_file), f'{expected_rdkit_raw_file} should exist'
    assert os.path.exists(expected_mordred_filtered_file), f'{expected_mordred_filtered_file} should exist'

    start = time.time()
    all_split_uuids = all_split_uuids + run_tests()
    end = time.time()
    second_run = end-start

    # loading features should make this run much faster
    assert second_run*2<first_run, f'Loading features should be at least twice as fast. Instead first run took {first_run} and second run took {second_run}'

    # clean
    shutil.rmtree(result_dir)
    os.remove(expected_rdkit_raw_file)
    os.remove(expected_mordred_filtered_file)

    for suuid in all_split_uuids:
        split_pattern = os.path.join(os.path.dirname(__file__), 
                         f'../../test_datasets/H1_hybrid*{suuid}.csv')
        splits = glob.glob(split_pattern)
        assert len(splits)==1, f'Should have found one split with {split_pattern}. Instead found {len(splits)}'

        os.remove(splits[0])

    print('cleaned and done')

if __name__ == '__main__':
    test_embedded_features()