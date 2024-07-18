import argparse
import os
import pytest
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
import atomsci.ddm.pipeline.parameter_parser as parse


config_path = currentdir + '/config_list_inputs.json'
filter_dict_path = currentdir + '/filter_dict_example.json'
required_inputs = ['--dataset_key','/ds/data/public/delaney/delaney-processed.csv',
                   '--bucket','gsk_ml']
wrong_type_inputs = ['--dataset_key','/ds/data/public/delaney/delaney-processed.csv',
                     '--bucket','5']
undefined_inputs = ['--dataset_key','/ds/data/public/delaney/delaney-processed.csv',
                    '--bucket','gsk_ml',
                    '--wrong_thing','1,2,3']
list_inputs = ['--dataset_key','/ds/data/public/delaney/delaney-processed.csv',
               '--bucket','gsk_ml',
               '--previously_split',
               '--model_type','NN,RF',
               '--batch_size','63',
               '--descriptor_type','moe',
               '--descriptor_key','/ds/projdata/gsk_data/GSK_Descriptors/all_GSK_Compound_2D_3D_MOE_Descriptors_Scaled_With_Smiles_And_Inchi.csv',
               '--response_cols','task1,task2,task3',
               '--hyperparam',
               '--dropouts','0.001,0.001 0.02,0.02,0.5 0.3,0.3,0.3',
               '--layer_sizes','1000,500 300,300,300 20000,50,50',
               '--learning_rate','1',
               '--weight_init_stddevs','0.001,0.001 0.02,0.02,0.5 0.3,0.3,0.3',
               '--bias_init_consts','1.0,1.0 2.0,2.0,2.0 0.3,0.3,0.3',
               '--model_filter',filter_dict_path]


dupe_inputs = ['--dataset_key','/ds/data/public/delaney/delaney-processed.csv',
               '--bucket','gsk_ml',
               '--dataset_key','nope']
config_inputs = ['--config_file', config_path, '--response_cols','one1,two2,three3','--previously_split','--baseline_epoch','78','--transformers','--splitter','random']

required_inputs_namespace = argparse.Namespace(dataset_key = '/ds/data/public/delaney/delaney-processed.csv', 
                                               bucket='gsk_ml')
undefined_inputs_namespace = argparse.Namespace(dataset_key = '/ds/data/public/delaney/delaney-processed.csv',
                                                bucket='gsk_ml',
                                                wrong_thing = '1,2,3')

# types are already pre-cast. 
# wrong_type_inputs_namespace = argparse.Namespace(dataset_key = '/ds/data/public/delaney/delaney-processed.csv',
#                                                 bucket=5)
list_inputs_namespace = argparse.Namespace(dataset_key ='/ds/data/public/delaney/delaney-processed.csv',
                                           bucket='gsk_ml',
                                           descriptor_type = 'moe', 
                                           descriptor_key = '/ds/projdata/gsk_data/GSK_Descriptors/all_GSK_Compound_2D_3D_MOE_Descriptors_Scaled_With_Smiles_And_Inchi.csv',
                                           layer_sizes = '42,42',
                                           batch_size = 63, 
                                           previously_split = True,
                                          response_cols = 'task1,task2,task3',
                                           hyperparam = True,
                                          transformers = False,
                                           model_type = 'NN,RF',
                                          model_filter = filter_dict_path)


required_inputs_dict = {'dataset_key' : '/ds/data/public/delaney/delaney-processed.csv',
                        'bucket':'gsk_ml'}
undefined_inputs_dict = {'dataset_key' : '/ds/data/public/delaney/delaney-processed.csv',
                         'bucket':'gsk_ml',
                         'wrong_thing' : '1,2,3'}
list_inputs_dict = {'dataset_key' : '/ds/data/public/delaney/delaney-processed.csv',
                    'bucket':'gsk_ml',
                    'descriptor_type' : 'moe',
                    'descriptor_key' : '/ds/projdata/gsk_data/GSK_Descriptors/all_GSK_Compound_2D_3D_MOE_Descriptors_Scaled_With_Smiles_And_Inchi.csv',
                    'layer_sizes' : '42,42',
                    'batch_size' : 63, 
                    'model_type' : 'NN,RF',
                    'previously_split' : True,
                    'hyperparam' : True,
                   'response_cols': 'task1,task2,task3',
                   'transformers': False,
                   'model_filter': filter_dict_path}
hierarchical_input_dict = {
    'ModelMetadata' : 
        {
        'TrainingDataset' : 
            {
                'dataset_key' : '/ds/data/public/delaney/delaney-processed.csv',
                'dataset_bucket' : 'gsk_ml',
                'descriptor_type' : 'moe',
                'descriptor_key' : 'None'
            },
        'ModelSpecific' : 
            {
                'system' : 'twintron-blue',
                'previously_split' : True
            }
        },
    'ModelTrainSelectionParameters' : 
        {
            'NNSpecific':
                {
                    'layer_sizes' : '42,42',
                    'batch_size' : 63,
                    'hyperparam' : 'True',
                    'response_cols': 'task1,task2,task3',
                    'transformers': 'False'
                },
    'IgnoreThisHeading' :
            {'model_type' : 'NN,RF'}
        }
}

command_line_namespace_inputs = [str(list_inputs_namespace)]
command_line_dict_inputs = [str(list_inputs_dict)]

"""# The file metadata_test.json is an example config file from the model zoo.
def test_model_metadata_input_as_dict():
    with open(currentdir + '/metadata_test.json') as f:
        config = json.loads(f.read())
    params = parse.wrapper(config)
    test.append(params.model_dataset_oid == '5bf33e862bf5f200d6201e59'
    test.append(params.transformers
    test.append(not params.uncertainty
    test.append(params.transformer_key == 'transformers_ed309e9f-b3b6-4111-88c7-3cbcccecf9c4.pkl'
    test.append(params.max_epochs == 3
    test.append(params.layer_sizes == [1000,500]
    test.append(params.result_dir == '/usr/local/data/delaney_refactored2'
"""

def test_default_params_json():
    params = parse.wrapper(currentdir + '/config_required_inputs.json')
    defaults = default_parameters()
    defaults.config_file = currentdir + '/config_required_inputs.json'
    test = []
    test.append(params == defaults)
    test.append(params.transformers)
    assert all(test)
    
def test_dupe_params_json(caplog):
    params = parse.wrapper(currentdir + '/config_dupe_inputs.json')
    for record in caplog.records:
        assert record.levelname == 'WARNING'
    
def test_incorrect_params_json(caplog):

   # with pytest.raises(ValueError):
    #with pytest.log(UserWarning):
    params = parse.wrapper(currentdir + '/config_wrong_inputs.json')
    for record in caplog.records:
        assert record.levelname == 'WARNING'
        

def test_correct_input_mixed_command_line_types():
    params = parse.wrapper(config_inputs)
    test = []
    test.append(params.system == 'twintron-blue')
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.layer_sizes == [[42, 42]])
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe') 
    test.append(not params.datastore)
    test.append(params.response_cols == ['one1','two2','three3'])
    test.append(params.baseline_epoch == 78)
    test.append(params.splitter == 'random')
    test.append(not params.transformers)
    assert all(test)
    
def test_correct_input_type_json():
    params = parse.wrapper(currentdir + '/config_list_inputs.json')
    test = []
    test.append(params.system == 'twintron-blue')
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.layer_sizes == [[42, 42]])
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe')
    test.append(not params.datastore)
    test.append(params.model_type == ['NN','RF'])
    test.append(params.response_cols == ['task1','task2','task3'])
    test.append(not params.transformers)
    assert all(test)

"""
#holdover test to ensure that parameters are being parsed
def test_params_exist():
    try:    
        params = parse.ParseParams.parse_command_line(required_inputs)
    except NameError:
        raise AssertionError
"""

def test_default_params_command():
    params = parse.wrapper(required_inputs)
    defaults = default_parameters()
    assert params == defaults
    

def test_default_params_command_with_dataset_hash():
    params = parse.wrapper(currentdir + '/../test_datasets/H1_hybrid.json')
    # a valid dataset hash should be generated after the parse.wrapper call
    assert params.dataset_hash != ''

"""
#test for ensuring system exits on required command
def test_required_vals_command():
    with pytest.raises(SystemExit):
        params = parse.ParseParams.parse_command_line()
"""


def test_undefined_param_command():
    with pytest.raises(SystemExit):
        params = parse.wrapper(undefined_inputs)
        
def test_dupe_param_command():
    with pytest.raises(ValueError):
        params = parse.wrapper(dupe_inputs)


        
def test_correct_input_type_command():

    params = parse.wrapper(list_inputs)
    test = []
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe' )
    test.append(params.response_cols == ['task1','task2','task3'])
    test.append(not params.datastore)
    test.append(params.model_type == ['NN','RF'])
    test.append(params.model_filter == {'model_uuid': 'uuid_1',
 'ModelMetadata.TrainingDataset.dataset_key': "['=', 2]",
 'ModelMetadata.TrainingDataset.dataset_bucket': "['>', 2]",
 'ModelMetadata.TrainingDataset.dataset_oid': "['>=', 2]",
 'ModelMetadata.TrainingDataset.class_names': "['in', [1,2,3]]",
 'ModelMetadata.TrainingDataset.num_classes': "['<', 2]",
 'ModelMetadata.TrainingDataset.feature_transform_type': "['<=', 2]",
 'ModelMetadata.TrainingDataset.response_transform_type': "['!=', 3]",
 'ModelMetadata.TrainingDataset.id_col': "['nin', [0,1,3,4]]"})
    test.append(params.dropouts == [[0.001,0.001], [0.02,0.02,0.5], [0.3,0.3,0.3]])
    test.append(params.layer_sizes == [[1000,500], [300,300,300], [20000,50,50]])
    test.append(params.weight_init_stddevs == [[0.001,0.001], [0.02,0.02,0.5], [0.3,0.3,0.3]])
    test.append(params.bias_init_consts == [[1.0,1.0], [2.0,2.0,2.0], [0.3,0.3,0.3]])
    test.append(params.learning_rate == 1)
    assert all(test)

def test_default_params_namespace():
    params = parse.wrapper(required_inputs_namespace)
    defaults = default_parameters()
    assert params == defaults
    params = parse.wrapper(required_inputs_dict)
    assert params == defaults
        
def test_required_vals_namespace(caplog):
    with pytest.raises(TypeError):
        parse.wrapper()

    


def test_undefined_param_namespace(caplog):
    params = parse.wrapper(undefined_inputs_namespace)
    for record in caplog.records:
        assert record.levelname == 'WARNING'
    params = parse.wrapper(undefined_inputs_dict)
    for record in caplog.records:
        assert record.levelname == 'WARNING'

    
def test_correct_input_type_namespace():

    params = parse.wrapper(list_inputs_namespace)
    test = []
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.layer_sizes == [[42, 42]])
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe')
    test.append(params.descriptor_key == '/ds/projdata/gsk_data/GSK_Descriptors/all_GSK_Compound_2D_3D_MOE_Descriptors_Scaled_With_Smiles_And_Inchi.csv')
    test.append(not params.datastore)
    test.append(params.response_cols == ['task1','task2','task3'])
    test.append(not params.transformers)
    test.append(params.model_type == ['NN','RF'])

    test.append(params.model_filter == {'model_uuid': 'uuid_1',
 'ModelMetadata.TrainingDataset.dataset_key': "['=', 2]",
 'ModelMetadata.TrainingDataset.dataset_bucket': "['>', 2]",
 'ModelMetadata.TrainingDataset.dataset_oid': "['>=', 2]",
 'ModelMetadata.TrainingDataset.class_names': "['in', [1,2,3]]",
 'ModelMetadata.TrainingDataset.num_classes': "['<', 2]",
 'ModelMetadata.TrainingDataset.feature_transform_type': "['<=', 2]",
 'ModelMetadata.TrainingDataset.response_transform_type': "['!=', 3]",
 'ModelMetadata.TrainingDataset.id_col': "['nin', [0,1,3,4]]"})
    
    
    params = parse.wrapper(list_inputs_dict)
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.layer_sizes == [[42, 42]])
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe')
    test.append(params.descriptor_key == '/ds/projdata/gsk_data/GSK_Descriptors/all_GSK_Compound_2D_3D_MOE_Descriptors_Scaled_With_Smiles_And_Inchi.csv')
    test.append(not params.datastore)
    test.append(params.model_type == ['NN','RF'])

    test.append(params.response_cols == ['task1','task2','task3'])
    test.append(not params.transformers)
    test.append(params.model_filter == {'model_uuid': 'uuid_1',
 'ModelMetadata.TrainingDataset.dataset_key': "['=', 2]",
 'ModelMetadata.TrainingDataset.dataset_bucket': "['>', 2]",
 'ModelMetadata.TrainingDataset.dataset_oid': "['>=', 2]",
 'ModelMetadata.TrainingDataset.class_names': "['in', [1,2,3]]",
 'ModelMetadata.TrainingDataset.num_classes': "['<', 2]",
 'ModelMetadata.TrainingDataset.feature_transform_type': "['<=', 2]",
 'ModelMetadata.TrainingDataset.response_transform_type': "['!=', 3]",
 'ModelMetadata.TrainingDataset.id_col': "['nin', [0,1,3,4]]"})
    assert all(test)
    
def test_command_line_namespace_and_dict_input():

    params = parse.wrapper(command_line_namespace_inputs)
    test = []
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.layer_sizes == [[42, 42]])
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe')
    test.append(params.descriptor_key == '/ds/projdata/gsk_data/GSK_Descriptors/all_GSK_Compound_2D_3D_MOE_Descriptors_Scaled_With_Smiles_And_Inchi.csv')
    test.append(not params.datastore)
    test.append(params.response_cols == ['task1','task2','task3'])
    test.append(not params.transformers)
    test.append(params.model_filter == {'model_uuid': 'uuid_1',
 'ModelMetadata.TrainingDataset.dataset_key': "['=', 2]",
 'ModelMetadata.TrainingDataset.dataset_bucket': "['>', 2]",
 'ModelMetadata.TrainingDataset.dataset_oid': "['>=', 2]",
 'ModelMetadata.TrainingDataset.class_names': "['in', [1,2,3]]",
 'ModelMetadata.TrainingDataset.num_classes': "['<', 2]",
 'ModelMetadata.TrainingDataset.feature_transform_type': "['<=', 2]",
 'ModelMetadata.TrainingDataset.response_transform_type': "['!=', 3]",
 'ModelMetadata.TrainingDataset.id_col': "['nin', [0,1,3,4]]"})
    params = parse.wrapper(command_line_dict_inputs)
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.layer_sizes == [[42, 42]])
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe')
    test.append(params.descriptor_key == '/ds/projdata/gsk_data/GSK_Descriptors/all_GSK_Compound_2D_3D_MOE_Descriptors_Scaled_With_Smiles_And_Inchi.csv')
    test.append(not params.datastore)
    test.append(params.response_cols == ['task1','task2','task3'])
    test.append(not params.transformers)
    test.append(params.model_filter == {'model_uuid': 'uuid_1',
 'ModelMetadata.TrainingDataset.dataset_key': "['=', 2]",
 'ModelMetadata.TrainingDataset.dataset_bucket': "['>', 2]",
 'ModelMetadata.TrainingDataset.dataset_oid': "['>=', 2]",
 'ModelMetadata.TrainingDataset.class_names': "['in', [1,2,3]]",
 'ModelMetadata.TrainingDataset.num_classes': "['<', 2]",
 'ModelMetadata.TrainingDataset.feature_transform_type': "['<=', 2]",
 'ModelMetadata.TrainingDataset.response_transform_type': "['!=', 3]",
 'ModelMetadata.TrainingDataset.id_col': "['nin', [0,1,3,4]]"})
    assert all(test)
    
def test_hierarchical_dict():
    params = parse.wrapper(hierarchical_input_dict)
    test = []
    test.append(params.system == 'twintron-blue')
    test.append(params.dataset_key == '/ds/data/public/delaney/delaney-processed.csv')
    test.append(params.layer_sizes == [[42, 42]])
    test.append(params.batch_size == 63)
    test.append(params.previously_split)
    test.append(params.descriptor_type == 'moe')
    test.append(not params.datastore)
    test.append(params.model_type == ['NN','RF'])
    test.append(params.response_cols == ['task1','task2','task3'])
    test.append(not params.transformers)
    assert all(test)


#test dictionary input
    
def default_parameters():
    default_params = parse.list_defaults()
    return default_params
