import atomsci.ddm.utils.generate_transformers as gt
import atomsci.ddm.pipeline.parameter_parser as pp
import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.transformations as trans
import atomsci.ddm.pipeline.compare_models as cm
import atomsci.ddm.utils.model_file_reader as mfr
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

import os
import shutil
import pandas as pd
import pickle as pkl
import tempfile
import pytest

def rel_path(filename):
    """
    Returns the absolute path to `filename` relative to the location of this file.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def clean(temp_dir, transformers_pkl_path):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    if os.path.exists(transformers_pkl_path):
        os.remove(transformers_pkl_path)

def test_transformer_generation():
    descriptor_type = 'rdkit_scaled'
    temp_root = rel_path('temp')
    transformers_pkl_path = rel_path('feature_transformers.pkl')
    clean(temp_root, transformers_pkl_path)
    assert not os.path.exists(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    prepared_H1 = gt.prepare_csv_and_descriptor_with_dummy_response(
        rel_path('data/H1_std_short.csv'),
        descriptor_type=descriptor_type,
        temp_root=temp_root,
        split_uuid='test-split'
    )

    assert os.path.exists(temp_root)
    assert os.path.exists(os.path.join(temp_root, 'H1_std_short.csv'))
    assert os.path.exists(os.path.join(temp_root, 'H1_std_short_train_valid_test_scaffold_test-split.csv'))
    assert 'dummy_response' in pd.read_csv(os.path.join(temp_root, 'H1_std_short.csv')).columns

    H1_config = {
                'dataset_key':prepared_H1,
                'previously_split':'True',
                'id_col':'compound_id',
                'smiles_col':'base_rdkit_smiles',
                'response_cols':'dummy_response',
                'split_uuid':'test-split',
                'splitter':'scaffold',
                }

    prepared_delaney = gt.prepare_csv_and_descriptor_with_dummy_response(
        rel_path('data/delaney-processed_curated_fit_short.csv'),
        descriptor_type=descriptor_type,
        temp_root=temp_root
    )

    assert os.path.exists(os.path.join(temp_root, 'scaled_descriptors/delaney-processed_curated_fit_short_with_rdkit_scaled_descriptors.csv'))

    delaney_config = {
                'dataset_key': prepared_delaney,
                'id_col':'Compound ID',
                'smiles_col':'smiles',
                'response_cols':'dummy_response',
            }

    aurk_config = {
                'dataset_key': rel_path('data/aurka_chembl_base_smiles_union_short.csv'),
                'id_col':'compound_id',
                'smiles_col':'base_rdkit_smiles',
                'response_cols':'pIC50',
            }

    transformer_dataset_key_configs = [H1_config, delaney_config, aurk_config]
    combined_dataset = gt.load_all_datasets(transformer_dataset_key_configs,
                                            featurizer='computed_descriptors',
                                            descriptor_type=descriptor_type)

    # aurka and delaney should have 200 each. H1 uses only 200 compounds from the training set
    # some compounds fail to featurize. There should be less than 600 compouds total
    assert len(combined_dataset) < 600

    # fit transformers
    gt.build_and_save_feature_transformers_from_csvs(
        transformer_dataset_key_configs,
        dest_pkl_path=transformers_pkl_path,
        featurizer='computed_descriptors',
        descriptor_type=descriptor_type,
        feature_transform_type='PowerTransformer',
        powertransformer_standardize='False',
    )

    with open(transformers_pkl_path, 'rb') as transformers_pkl_file:
        saved_transformers = pkl.load(transformers_pkl_file)

    assert 'transformers_x' in saved_transformers
    assert 'params' in saved_transformers

    assert transformer_dataset_key_configs == saved_transformers['params']['transformer_dataset_key_configs']
    assert saved_transformers['params']['featurizer'] == 'computed_descriptors'
    assert saved_transformers['params']['descriptor_type'] == descriptor_type
    assert saved_transformers['params']['feature_transform_type'] == 'PowerTransformer'

    # test load transformers
    # first create a params json this uses a RobustScaler, we will check
    # that the PowerTransformer is used instead
    params_json = {
        "dataset_key" : rel_path('data/H1_std_short.csv'),
        "datastore" : "False",
        "uncertainty": "False",
        "splitter": "scaffold",
        "previously_split": "True",
        "split_uuid": "test-split",
        "prediction_type": "regression",
        "response_cols" : "pKi_mean",
        "id_col": "compound_id",
        "smiles_col" : "base_rdkit_smiles",
        "result_dir": temp_root,
        "model_type": "RF",
        "featurizer": "ecfp",
        "descriptor_type": 'rdkit_raw',
        "feature_transform_type": "RobustScaler",
        "feature_transform_path": transformers_pkl_path,
        "save_results": "False",
        "verbose": "False",
        "seed":"0"
    }

    # this one will fail because it has the wrong featurizer
    with pytest.raises(ValueError) as excinfo:
        bad_params1 = pp.wrapper(params_json)
        model_pipeline = mp.ModelPipeline(bad_params1)
        model_pipeline.train_model()
    assert 'Loaded transformers do not match featurizer.' in str(excinfo.value)

    # This one will fail because it ahas the wrong descriptor type
    params_json['featurizer'] = 'computed_descriptors'
    with pytest.raises(ValueError) as excinfo:
        bad_params2 = pp.wrapper(params_json)
        model_pipeline = mp.ModelPipeline(bad_params2)
        model_pipeline.train_model()
    assert 'Loaded transformers do not match descriptor_type.' in str(excinfo.value)

    # This one will succeed.
    params_json['descriptor_type'] = descriptor_type
    test_params = pp.wrapper(params_json)
    model_pipeline = mp.ModelPipeline(test_params)
    model_pipeline.train_model()

    # check that the PowerTransformer was used instead of RobustScaler
    assert model_pipeline.params.feature_transform_type == 'PowerTransformer'
    transformers_x = model_pipeline.model_wrapper.transformers_x
    assert len(transformers_x[0])==1
    assert isinstance(transformers_x[0][0], trans.SklearnPipelineWrapper)
    assert isinstance(transformers_x[0][0].sklearn_pipeline, Pipeline)
    scaler = transformers_x[0][0].sklearn_pipeline.named_steps['PowerTransformer']
    assert isinstance(scaler, PowerTransformer)

    # check that the saved transformer pkl is a PowerTransformer
    # reload the pipeline and see that it was loaded correctly works
    results = cm.get_filesystem_perf_results(temp_root, pred_type='regression')
    model_path = results['model_path'].values[0]
    pred_params = {
        'featurizer': 'computed_descriptors',
        'result_dir': tempfile.mkdtemp(),
        'id_col': 'compound_id',
        'smiles_col':'base_rdkit_smiles' 
    }
    pred_params = pp.wrapper(pred_params)
    reloaded_pipeline = mp.create_prediction_pipeline_from_file(pred_params, reload_dir=None, model_path=model_path)

    assert reloaded_pipeline.params.feature_transform_type == 'PowerTransformer'
    rl_formers_x = reloaded_pipeline.model_wrapper.transformers_x
    scaler = rl_formers_x[0][0].sklearn_pipeline.named_steps['PowerTransformer']
    assert isinstance(scaler, PowerTransformer)

    # check that model metadata shows that this was a PowerTransformer
    # check that standardize is set to false
    reader = mfr.ModelFileReader(model_path)
    assert not reader.get_powertransformer_standardize()

    # check that the dataset_key configs were also saved in the parameters
    loaded_trans_dskey_configs = reader.get_transformer_dataset_key_configs()
    assert loaded_trans_dskey_configs == transformer_dataset_key_configs


    # cleanup
    clean(temp_root, transformers_pkl_path)

if __name__ == '__main__':
    test_transformer_generation()