import os
import pytest
import atomsci.ddm.pipeline.featurization as feat
import atomsci.ddm.pipeline.parameter_parser as parse
import deepchem as dc
from deepchem.models import GraphConvModel
import atomsci.ddm.pipeline.model_datasets as model_dataset
import atomsci.ddm.pipeline.model_wrapper as model_wrapper
import atomsci.ddm.pipeline.model_pipeline as MP

import utils_testing as utils
import copy

"""This testing script assumes that /ds/data/public/delaney/delaney-processed.csv is still on the same path on twintron. Assumes that the dataset_key: /ds/projdata/gsk_data/GSK_derived/PK_parameters/gsk_blood_plasma_partition_rat_crit_res_data.csv under the bucket gskdata and with the object_oid: 5af0e6368003ff018de33db5 still exists. 
"""

#The dataset object from file is a delaney dataset using an ecfp featurizer with a default scaffold split.

datastore_is_down = utils.datastore_status()
MP_delaney_ecfp_train_valid_test_random = utils.delaney_pipeline()
(delaney_params, mdl_dataset_delaney, delaney_df) = utils.delaney_objects()

general_params = {'dataset_key' : './delaney-processed.csv',
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





DD = dc.data.datasets.NumpyDataset

#***********************************************************************************
def test_create_model_wrapper():
    """
        Args:
        params (Namespace) : Parameters passed to the model pipeline
                     featurizer (Featurization): Object managing the featurization of compounds
                                 ds_client (DatastoreClient): Interface to the file datastore

                                                  Returns:
                                                  model (pipeline.Model): Wrapper for DeepChem, sklearn or other model.

                                                              Raises:
ValueError: Only params.model_type = 'NN' or 'RF' is supported. 

Dependencies:
None

Calls:
MultitaskDCModelWrapper, DCRFModelWrapper
    """
    inp_params = parse.wrapper(general_params)
    featurization = feat.create_featurization(inp_params)
    mdl = model_wrapper.create_model_wrapper(inp_params, featurization)
    mdl.setup_model_dirs()
    
    # testing for correct attribute initialization with model_type == "NN"
    test = []
    test.append(mdl.params.model_type == 'NN')
    test.append(isinstance(mdl.featurization,feat.DynamicFeaturization))
    test.append(mdl.output_dir == inp_params.output_dir)
    test.append(mdl.model_dir == inp_params.output_dir + '/' + 'model')
    test.append(mdl.best_model_dir == inp_params.output_dir + '/' + 'best_model')
    test.append(mdl.transformers == [])
    test.append(mdl.transformers_x == [])
    test.append(isinstance(mdl, model_wrapper.MultitaskDCModelWrapper))

    # testing for correct attribute initialization with model_type == "RF"
    temp_params = copy.deepcopy(inp_params)
    temp_params.model_type = 'RF'
    featurization = feat.create_featurization(temp_params)
    mdl_RF = model_wrapper.create_model_wrapper(temp_params, featurization)
    test.append(isinstance(mdl_RF, MP.model_wrapper.DCRFModelWrapper))
    test.append(mdl_RF.params.model_type == 'RF')

    # assertion for all tests
    assert all(test)

    #testing for Exception with model_type not in ['NN','RF']
    with pytest.raises(ValueError):
        temp_params.model_type = 'wrong'
        mdl_wrong = model_wrapper.create_model_wrapper(temp_params, featurization)


#***********************************************************************************
def test_super_create_transformers():
    """Args:
    model_dataset: The ModelDataset object that handles the current dataset

    Returns:
    self.transformers
    self.transformers_x
    self.params.transformer_key
    self.params.transformer_oid (if datastore)

    Raises:
    Exception when failing to save to the datastore

    Dependencies:
    create_featurization
    create_model_dataset
    model_dataset.load_full_dataset
    model_dataset.get_dataset_tasks
    model_dataset.check_task_columns
    model_dataset.get_featurized_data
    Requires (self.params.prediction_type == 'regression' and self.params.transformers == True) or len(self.transformers) > 0

    Calls:
    self.featurization.create_feature_transformer
    dsf.upload_pickle_to_DS

    """
    #set up for a model wrapper with regression and NN.

    inp_params = parse.wrapper(general_params)
    featurization = feat.create_featurization(inp_params)
    data_obj_ecfp = model_dataset.create_model_dataset(inp_params, featurization, ds_client = None)
    df_delaney = data_obj_ecfp.load_full_dataset()
    data_obj_ecfp.get_dataset_tasks(df_delaney)
    model_dataset.check_task_columns(inp_params, df_delaney)
    data_obj_ecfp.get_featurized_data()
    mdl = model_wrapper.create_model_wrapper(inp_params, data_obj_ecfp.featurization)
    mdl.setup_model_dirs()

    #testing correct model_wrapper build with regression and NN
    test = []
    test.append(mdl.params.prediction_type == 'regression')
    test.append(mdl.params.model_type == 'NN')
    mdl.create_transformers(data_obj_ecfp)
    test.append(isinstance(mdl.transformers[0], dc.trans.transformers.NormalizationTransformer))
    test.append(mdl.transformers_x == [])
    #testing saving of transformer to correct location:
    transformer_path = os.path.join(mdl.output_dir, 'transformers.pkl')
    test.append(os.path.isfile(transformer_path))

    # TODO: test proper saving of the transformer to the datastore

    # TODO: test when transformers is False:
    inp_params.prediction_type = 'classification'
    mdl = model_wrapper.create_model_wrapper(inp_params, featurization)
    test.append(mdl.transformers == [])
    test.append(mdl.transformers_x == [])
    assert all(test)


#***********************************************************************************
def test_super_transform_dataset():
    """Args:
    dataset: The DeepChem DiskDataset that contains a dataset

    Returns:
    transformed_dataset

    Raises:
    None

    Dependencies:
    model_dataset.create_transformers

    Calls:
    None

    """
    #set up for a model wrapper with regression and NN.
    inp_params = parse.wrapper(general_params)
    featurization = feat.create_featurization(inp_params)
    data_obj_ecfp = model_dataset.create_model_dataset(inp_params, featurization, ds_client = None)
    df_delaney = data_obj_ecfp.load_full_dataset()
    data_obj_ecfp.get_dataset_tasks(df_delaney)
    model_dataset.check_task_columns(inp_params, df_delaney)
    data_obj_ecfp.get_featurized_data()
    mdl = model_wrapper.create_model_wrapper(inp_params, data_obj_ecfp.featurization)
    mdl.setup_model_dirs()
    mdl.create_transformers(data_obj_ecfp)
    dataset = mdl.transform_dataset(data_obj_ecfp.dataset)

    test = []
    # checking that the dataset is the correct type
    test.append(isinstance(dataset, DD))
    # since this is not descriptor featurization, the X values for the datasets should be the same
    test.append((dataset.X == data_obj_ecfp.dataset.X).all())
    # and the response values should be the same length:
    test.append(len(dataset.y) == len(data_obj_ecfp.dataset.y))
    test.append(len(dataset.y) == len(dataset.ids))
    assert all(test)




#***********************************************************************************
#def test_train_NN_graphconv_scaffold_defaults():
#    """
#
#    Args:
#    pipeline (ModelPipeline): The ModelPipeline instance for this model run.
#    
#    Dependencies:
#    ModelPipeline creation
#    featurization creation
#    creation of model_wrapper
#    mp.load_featurize_data
#
#    Calls:
#    create_perf_data
#    perf_data.accumulate_preds
#    perf_data.comput_perf_metrics
#    data.combined_training-data()
#    self._copy_model
#    """    
#    # set up for the default graphconv model.
#    general_params['featurizer'] = 'graphconv'
#    inp_params = parse.wrapper(general_params)
#    mp = MP.ModelPipeline(inp_params)
#    mp.featurization = feat.create_featurization(inp_params)
#    mp.model_wrapper = model_wrapper.create_model_wrapper(inp_params, mp.featurization, mp.ds_client)
#    # asserting that the default model is created with the correct layer sizes, dropouts, model_dir, and mode
#    test2 = []
#    test2.append(mp.model_wrapper.params.layer_sizes == [64, 64, 128])
#    print(mp.model_wrapper.params.dropouts)
#    test2.append(mp.model_wrapper.params.dropouts == [0.25,0.25,0.25])
#    test2.append(isinstance(mp.model_wrapper.model, dc.models.tensorgraph.models.graph_models.GraphConvModel))
#    test2.append(mp.model_wrapper.model.model_dir == mp.model_wrapper.model_dir)
#    test2.append(mp.model_wrapper.model.graph_conv_layers == [64,64])
#    test2.append(mp.model_wrapper.model.dropout == [0.25,0.25,0.25])
#    test2.append(mp.model_wrapper.model.mode == 'regression')
#    test2.append(mp.model_wrapper.model.dense_layer_size == 128)
#    assert all(test2)
#    
#    # calling dependencies for model_wrapper.train()
#    mp.model_wrapper.setup_model_dirs()
#    mp.load_featurize_data()
#    mp.model_wrapper.train(mp)
#
#    mdl = mp.model_wrapper
#    test3 = []
#    #asserting that attributes are properly created during train
#    test3.append(isinstance(mdl.data, model_dataset.FileDataset))
#    test3.append(isinstance(mdl.best_epoch, int))
#    test3.append(isinstance(mdl.train_perf_data, list))
#    test3.append(isinstance(mdl.train_perf_data[-1],perf_data.SimpleRegressionPerfData))
#    test3.append(isinstance(mdl.valid_perf_data, list))
#    test3.append(isinstance(mdl.valid_perf_data[-1],perf_data.SimpleRegressionPerfData))
#    test3.append(isinstance(mdl.train_epoch_perfs, np.ndarray))
#    test3.append(isinstance(mdl.valid_epoch_perfs, np.ndarray))
#    assert all(test3)
#
#    # For this particular dataset with graphconv and scaffold splitting, the training r2 tends to reach 0.4 by 10 epochs, setting the threshold to 0.3. If the training r2 is less than 0.4, there is most likely something wrong with the model training.
#    assert max(mdl.train_epoch_perfs) > 0.3
#    # For this particular dataset with graphconv, the valid r2 tends to reach 0.1 by 10 epochs
#    assert max(mdl.valid_epoch_perfs) > 0.1
#
#    # checking that the real_vals within train_perf_data and valid_perf_data match the datasets they originated from.
#    test4 = []
#    test4.append((mp.data.train_valid_dsets[0][0].y == mp.model_wrapper.train_perf_data[-1].real_vals).all())
#    test4.append((mp.data.train_valid_dsets[0][1].y == mp.model_wrapper.valid_perf_data[-1].real_vals).all())
#    # checking that the ids match as well
#    test4.append((mp.data.train_valid_dsets[0][0].ids == mp.model_wrapper.train_perf_data[-1].ids).all())
#    test4.append((mp.data.train_valid_dsets[0][1].ids == mp.model_wrapper.valid_perf_data[-1].ids).all())
#    # checking that the predicted values have the correct length and type
#    test4.append(len(mp.model_wrapper.train_perf_data[-1].pred_vals) == len(mp.data.train_valid_dsets[0][0].y))
#    test4.append(isinstance(mp.model_wrapper.train_perf_data[-1].pred_vals[0][0], np.float32))
#    test4.append(len(mp.model_wrapper.valid_perf_data[-1].pred_vals) == len(mp.data.train_valid_dsets[0][1].y))
#    test4.append(isinstance(mp.model_wrapper.valid_perf_data[-1].pred_vals[0][0], np.float32))
#    assert all(test4)

#***********************************************************************************
def test_train_NN_graphconv_scaffold_inputs():
    """Args:
    pipeline (ModelPipeline): The ModelPipeline instance for this model run.

    Dependencies:
    ModelPipeline creation
    featurization creation
    creation of model_wrapper
    mp.load_featurize_data

    Calls:
    create_perf_data
    perf_data.accumulate_preds
    perf_data.comput_perf_metrics
    data.combined_training-data()
    self._copy_model
    """
    # checking that the layers, dropouts, and learning rate are properly added to the deepchem graphconv model
    general_params['featurizer'] = 'graphconv'
    general_params['layer_sizes'] = '100,100,10'
    general_params['dropouts'] = '0.3,0.3,0.1'
    general_params['uncertainty'] = False
    inp_params = parse.wrapper(general_params)
    mp = MP.ModelPipeline(inp_params)
    mp.featurization = feat.create_featurization(inp_params)
    mp.model_wrapper = model_wrapper.create_model_wrapper(inp_params, mp.featurization, mp.ds_client)
    # asserting that the correct model is created with the correct layer sizes, dropouts, model_dir, and mode by default
    test1 = []
    
    test1.append(mp.model_wrapper.params.layer_sizes == [100, 100, 10])
    test1.append(mp.model_wrapper.params.dropouts == [0.3,0.3,0.1])
    # checking that parameters are properly passed to the deepchem model object
    test1.append(isinstance(mp.model_wrapper.model, GraphConvModel))
    test1.append(mp.model_wrapper.model.model_dir == mp.model_wrapper.model_dir)
    test1.append([i.out_channel for i in mp.model_wrapper.model.model.graph_convs]== [100,100])
    test1.append([i.rate for i in mp.model_wrapper.model.model.dropouts] == [0.3,0.3,0.1])
    test1.append(mp.model_wrapper.model.mode == 'regression')
    test1.append(mp.model_wrapper.model.model.dense.units == 10)
    assert all(test1)
    
    #***********************************************************************************
    def test_super_get_train_valid_pred_results():
        """Args:
        perf_data: A PerfData object that stores the predicted values and metrics
        Returns:
        dict: A dictionary of the prediction results

            Raises:
        None

        Dependencies:
        create_perf_data

        Calls:
        perf_data.get_prediction_results()

        """
        pass
    # should be tested in perf_data.get_prediction_results()
    # should still be called to make sure that the function is callable

    #***********************************************************************************
    def test_super_get_test_perf_data():
        """Args:
        model_dir (str): Directory where the saved model is stored
        model_dataset (DiskDataset): Stores the current dataset and related methods

        Returns:
        perf_data: PerfData object containing the predicted values and metrics for the current test dataset

            Raises:
        None

        Dependencies:
        A model must be in model_dir
        model_dataset.test_dset must exist

        Calls:
        create_perf_data
        self.generate_predictions
        perf_data.accumulate_preds
        """
        pass
        # mostly tested in accumulate_preds, but should be tested to ensure taht the predictions are properly being called

    #***********************************************************************************
    def test_super_get_test_pred_results():
        """Args:
        model_dir (str): Directory where the saved model is stored
        model_dataset (DiskDataset): Stores the current dataset and related methods

        Returns:
        dict: A dictionary containing the prediction values and metrics for the current dataset.

            Raises:
        None

        Dependencies:
        A model must be in model_dir
        model_dataset.test_dset must exist

        Calls:
        self.get_test_perf_data
        perf_data.get_prediction_results
        """
        pass
        #mostly tested in perf_data.get_prediction_results

    #***********************************************************************************
    def test_super_get_full_dataset_perf_data():
        """Args:
        model_dataset (DiskDataset): Stores the current dataset and related methods

        Returns:
        perf_data: PerfData object containing the predicted values and metrics for the current full dataset

            Raises:
        None

        Dependencies:
        A model must already be trained

        Calls:
        create_perf_data
        self.generate_predictions
        self.accumulate_preds
        """
        pass

    #***********************************************************************************
    def test_super_get_full_dataset_pred_results():
        """Args:
        model_dataset (DiskDataset): Stores the current dataset and related methods
        Returns:
        dict: A dictionary containing predicted values and metrics for the current full dataset

            Raises:
        None

        Dependencies:
        A model was already be trained.

        Calls:
        get_full_dataset_perf_data
        self.get_prediction_results()
        """
        pass
