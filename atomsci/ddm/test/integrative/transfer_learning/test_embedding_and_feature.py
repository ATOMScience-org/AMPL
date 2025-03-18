# This tests using an existing NN model to 
# generate embedding features.
import os
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse
import atomsci.ddm.pipeline.compare_models as cp
import atomsci.ddm.utils.model_file_reader as mfr
import shutil
import glob

def test_embedded_features():

    dskey = os.path.join(os.path.dirname(__file__), 
                         '../../test_datasets/H1_hybrid.csv')

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
                "embedding_and_features": True,
                "learning_rate": .0001,
                "layer_sizes": [64, 64, 64],
                "dropouts": [0.2, 0.2, 0.2],
                "max_epochs": "5",
                "early_stopping_patience": "5",
                "verbose": "False"
            }

            pparams = parse.wrapper(transfer_json)

            model_pipeline = mp.ModelPipeline(pparams)
            model_pipeline.train_model()

            # test that there are indeed more features
            assert model_pipeline.data.featurization.get_feature_count() > model_pipeline.data.featurization.input_featurization.get_feature_count()

            split_uuids.append(model_pipeline.data.split_uuid)

            model_uuid = model_pipeline.params.model_uuid
            results_df = cp.get_filesystem_perf_results(result_dir=result_dir, pred_type='regression')
            trained_model_path = results_df[results_df['model_uuid']==model_uuid]['model_path'].values[0]

            reader = mfr.ModelFileReader(trained_model_path)

            assert reader.get_embedding_model_path() == model_path
            assert reader.get_embedding_and_features()

        return split_uuids

    all_split_uuids = run_tests()
    
    #clean
    shutil.rmtree(result_dir)
    for suuid in all_split_uuids:
        split_pattern = os.path.join(os.path.dirname(__file__), 
                         f'../../test_datasets/H1_hybrid*{suuid}.csv')
        splits = glob.glob(split_pattern)
        assert len(splits)==1, f'Should have found one split with {split_pattern}. Instead found {len(splits)}'

        os.remove(splits[0])

    print('cleaned and done')


if __name__ == '__main__':
    test_embedded_features()