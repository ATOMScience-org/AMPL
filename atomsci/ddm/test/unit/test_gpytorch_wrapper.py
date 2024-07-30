from atomsci.ddm.pipeline import featurization, parameter_parser, model_pipeline, model_wrapper
#from atomsci.ddm.pipeline.model_wrapper import GPyTorchModelWrapper

general_params = {'dataset_key' : '../test_datasets/delaney-processed_curated_fit.csv',
'featurizer': 'ecfp',
'response_cols': 'measured log solubility in mols per litre',
'id_col': 'Compound ID',
'smiles_col': 'smiles',
'output_dir': 'pytest',
'model_type' : 'NN',
'splitter' : 'scaffold',
'prediction_type' : 'regression',
'baseline_epoch' : '7',
'max_epochs' : '10',
'datastore': 'False',
'save_results': 'False'}

mp = model_pipeline.ModelPipeline(parameter_parser.wrapper(general_params))
mp.featurization = featurization.create_featurization(mp.params)
mp.model_wrapper = model_wrapper.create_model_wrapper(mp.params, mp.featurization, mp.ds_client)
mp.load_featurize_data()
test_object = model_wrapper.GPyTorchModelWrapper(params=mp.params, featurizer=mp.featurization, ds_client=None)
test_object.train(mp)

