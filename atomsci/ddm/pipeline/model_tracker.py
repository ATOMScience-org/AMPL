"""
Module to interface model pipeline to model tracker service.
"""

import os
import subprocess
import tempfile
import sys
import pandas as pd

from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import mlmt_client_wrapper as mlmt_client_wrapper

class UnableToTarException(Exception):
    pass

class DatastoreInsertionException(Exception):
    pass
class MLMTClientInstantiationException(Exception):
    pass

# *********************************************************************************************************************************
def save_model(pipeline, collection_name='model_tracker', log=True):
    """Save the model.

    Save the model files to the datastore and save the model metadata dict to the Mongo database.

    Args:
        pipeline (ModelPipeline object): the pipeline to use
        collection_name (str): the name of the Mongo DB collection to use
        log (bool): True if logs should be printed, default False
        use_personal_client (bool): True if personal client should be used (i.e. for testing), default False

    Returns:
        None if insertion was successful, raises UnableToTarException, DatastoreInsertionException, MLMTClientInstantiationException
        or MongoInsertionException otherwise
    """
    
    if pipeline is None:
        raise Exception('pipeline cannot be None.')

    # ModelPipeline.create_model_metadata() should be called before the call to save_model.
    # Get the metadata dictionary from the model pipeline.
    metadata_dict = pipeline.model_metadata
    model_uuid = metadata_dict['model_uuid']
    if model_uuid is None:
        raise ValueError("model_uuid is missing from pipeline metadata.")

    #### Part 1: Save the model tarball ####
    model = pipeline.model_wrapper
    # best_model_dir is an absolute path.
    directory_to_tar = model.best_model_dir
    # Put tar file in a temporary directory that will automatically be destroyed when we're done
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_file = os.path.join(tmp_dir, 'model_{model_uuid}.tar.gz'.format(model_uuid=model_uuid))
        tar_flags = 'czf'
        # Change directory to model_dir so that paths in tarball are relative to model_dir.
        tar_command = 'tar -{tar_flags} {tar_file} -C {directory_to_tar} .'.format(tar_flags=tar_flags, tar_file=tar_file,
                                                                                   directory_to_tar=directory_to_tar)
        try:
            subprocess.check_output(tar_command.split())
        except subprocess.CalledProcessError as e:
            pipeline.log.error('Command to create model tarball returned status {return_code}'.format(return_code=e.returncode))
            pipeline.log.error('Command was: "{cmd}"'.format(cmd=e.cmd))
            pipeline.log.error('Output was: "{output}"'.format(output=e.output))
            pipeline.log.error('stderr was: "{stderr}"'.format(stderr=e.stderr))
            raise UnableToTarException('Unable to tar {directory_to_tar}.'.format(directory_to_tar=directory_to_tar))
        title = '{model_uuid} model tarball'.format(model_uuid=model_uuid)
        uploaded_results = dsf.upload_file_to_DS(
            bucket=pipeline.params.model_bucket, title=title, description=title, tags=[],
            key_values={'model_uuid' : model_uuid, 'file_category': 'ml_model'}, filepath=tmp_dir,
            filename=tar_file, dataset_key='model_' + model_uuid + '_tarball', client=pipeline.ds_client,
            return_metadata=True)
        if uploaded_results is None:
            raise DatastoreInsertionException('Unable to upload title={title} to datastore.'.format(title=title))
    # Get the dataset_oid for actual metadata file stored in datastore.
    model_dataset_oid = uploaded_results['dataset_oid']
    # By adding dataset_oid to the dict, we can immediately find the datastore file asssociated with a model.
    metadata_dict['ModelMetadata']['ModelParameters']['model_dataset_oid'] = model_dataset_oid


    #### Part 2: Save the model metadata ####
    client_wrapper = mlmt_client_wrapper.MLMTClientWrapper()
    client_wrapper.instantiate_mlmt_client()
    if client_wrapper.mlmt_client is None:
        raise MLMTClientInstantiationException(
            'mlmt_client failed to instantitate.')
    # Temporarily add collection_name key. The model tracker will use this key internally and pop it from the dict.
    metadata_dict['collection_name'] = collection_name
    client_wrapper.save_metadata(model_metadata_dict=metadata_dict)
    if log:
        print('Successfully inserted into the database with model_uuid %s.' % model_uuid)

# *********************************************************************************************************************************
def get_models(filter_dict, client_wrapper=None, collection_name='model_tracker', log=False):
    """Retrieve relevant models.

    Retrieve models matching given criteria.

    Args:
        filter_dict (dict): dictionary to filter on

    Returns:
        A list of matching model dictionaries (matching metadata with matching
        metrics). Raises MongoQueryException if the query fails.
    """
    if filter_dict is None:
        raise Exception('filter_dict cannot be None.')
    if client_wrapper is None:
        client_wrapper = mlmt_client_wrapper.MLMTClientWrapper(ds_client=dsf.config_client())
        client_wrapper.instantiate_mlmt_client()

    # Temporarily add collection_name key. The model tracker will use this key
    # internally and pop it from the dict.
    filter_dict['collection_name'] = collection_name
    gen = client_wrapper.get_models_generator(filter_dict=filter_dict, log=log)
    if log:
        print('Successfully constructed models generator.')
    return gen

# *********************************************************************************************************************************
def get_full_metadata(filter_dict, client_wrapper=None, collection_name='model_tracker', log=False):
    """Retrieve relevant full metadata (including TrainingRun metrics).

    Retrieve full metadata of models matching given criteria.

    Args:
        filter_dict (dict): dictionary to filter on

    Returns:
        A list of matching full model metadata (including TrainingRun metrics) dictionaries. Raises MongoQueryException if the query fails.
    """
    if filter_dict is None:
        raise Exception('filter_dict cannot be None.')
    if client_wrapper is None:
        client_wrapper = mlmt_client_wrapper.MLMTClientWrapper(ds_client=dsf.config_client())
        client_wrapper.instantiate_mlmt_client()

    # Temporarily add collection_name key. The model tracker will use this key
    # internally and pop it from the dict.
    filter_dict['collection_name'] = collection_name
    gen = client_wrapper.get_full_metadata_generator(filter_dict=filter_dict, log=log)
    if log:
        print('Successfully constructed models generator.')
    return gen


# *********************************************************************************************************************************

def get_metadata(filter_dict, client_wrapper=None, collection_name='model_tracker',
               log=False):
    """Retrieve relevant metadata.

    Retrieve metadata matching given criteria.

    Args:
        filter_dict (dict): dictionary to filter on

    Returns:
        A list of matching metadata dictionaries. Raises MongoQueryException if
        the query fails.
    """

    if filter_dict is None:
        raise Exception('filter_dict cannot be None.')
    if client_wrapper is None:
        client_wrapper = mlmt_client_wrapper.MLMTClientWrapper(ds_client=dsf.config_client())
        client_wrapper.instantiate_mlmt_client()

    # Temporarily add collection_name key. The model tracker will use this key
    # internally and pop it from the dict.
    filter_dict['collection_name'] = collection_name
    gen = client_wrapper.get_metadata_generator(filter_dict=filter_dict, log=log)
    if log:
        print('Successfully constructed metadata generator.')
    return gen

# *********************************************************************************************************************************

def get_metadata_by_uuid(uuid, client_wrapper=None, collection_name=None, log=False):
    """Retrieve relevant model metadata by uuid.

    Retrieve metadata matching given uuid

    Args:
        uuid (str): model uuid
        collection(str): collection to search (optional, searches all collections if not specified)
    Returns:
        Matching metadata dictionary. Raises MongoQueryException if the query fails.
    """
    
    if client_wrapper is None:
        client_wrapper = mlmt_client_wrapper.MLMTClientWrapper(ds_client=dsf.config_client())
        client_wrapper.instantiate_mlmt_client()
        
    if not client_wrapper.mlmt_client:
        raise Exception('mlmt_client failed to instantitate')
    
    if collection_name is None:
        collection_name = get_model_collection_by_uuid(uuid, client_wrapper=client_wrapper)
        
    model_meta = list(get_full_metadata({"model_uuid" : uuid}, client_wrapper=client_wrapper, collection_name=collection_name))

    return model_meta[0]

# *********************************************************************************************************************************
def get_model_collection_by_uuid(uuid, client_wrapper=None):
    """Retrieve model collection given a uuid.

    Retrieve model collection given a uuid.

    Args:
        uuid (str): model uuid
    Returns:
        Matching collection name
    """
    
    if client_wrapper is None:
        client_wrapper = mlmt_client_wrapper.MLMTClientWrapper(ds_client=dsf.config_client())
        client_wrapper.instantiate_mlmt_client()
        
    collection = 'Collection not found for uuid: ' + uuid
    colls = client_wrapper.get_collection_names({})   
    for col in colls['matching_collection_names']:
        model_meta = list(get_full_metadata({"model_uuid" : uuid}, client_wrapper=client_wrapper,collection_name=col))
        if model_meta != []:
            collection = col
            break
    
    return collection

# *********************************************************************************************************************************
def get_model_training_data_by_uuid(uuid):
    """Retrieve data used to train, validate, and test a model given the uuid

    Args:
        uuid (str): model uuid
    Returns:
        a tuple of datafraes containint training data, validation data, and test data including the compound ID, RDKIT SMILES, and response value
    """
    model_meta = get_metadata_by_uuid(uuid)
    response_col = model_meta['ModelMetadata']['TrainingDataset']['response_cols']
    smiles_col = model_meta['ModelMetadata']['TrainingDataset']['smiles_col']
    full_data  = dsf.retrieve_dataset_by_dataset_oid(model_meta['ModelMetadata']['TrainingDataset']['dataset_oid'], verbose=False)

    # Pull split data and merge into initial dataset
    split_meta = dsf.search_datasets_by_key_value('split_dataset_uuid', model_meta['ModelMetadata']['SplittingParameters']['Splitting']['split_uuid'])
    split_oid  = split_meta['dataset_oid'].values[0]
    split_data = dsf.retrieve_dataset_by_dataset_oid(split_oid, verbose=False)
    split_data['compound_id'] = split_data['cmpd_id']
    split_data = split_data.drop(columns=['cmpd_id'])
    full_data = pd.merge(full_data, split_data, how='inner', on=['compound_id'])

    train_data = full_data[full_data['subset'] == 'train'][['compound_id',smiles_col,*response_col]].reset_index(drop=True)
    valid_data = full_data[full_data['subset'] == 'valid'][['compound_id',smiles_col,*response_col]].reset_index(drop=True)
    test_data  = full_data[full_data['subset'] == 'test'][['compound_id',smiles_col,*response_col]].reset_index(drop=True)

    return train_data, valid_data, test_data


# *********************************************************************************************************************************
def save_metrics(pipeline, model_metrics, collection_name='model_tracker', log=False):
    """Wrapper for mlmt_client save_model_metrics() function. Stores the performance metrics from a training
    or prediction run in the model tracker database.

    Args:
        pipeline (ModelPipeline): The ModelPipeline object managing the training or prediction run.

        model_metrics (dict): A dictionary containing the performance metrics data to be stored. It should have one of the following forms:
            {'model_uuid' : <model_UUID>, 'ModelMetrics' : {'TrainingRun' : {'label' : label, 'subset' : subset, 'PredictionResults' : results}}
            {'model_uuid' : <model_UUID>, 'ModelMetrics' : {'PredictionRuns' : {'time_run' : time_run, 'dataset_oid' : dataset_oid, <etc.>,
                                                                           'PredictionResults' : results}}
        The dict will be appended to the appropriate model's list of metrics dicts in the model tracker DB.

    Returns:
        None if insertion was successful, raises MLMTClientInstantiationException or MongoInsertionException otherwise.
    """

    client_wrapper = pipeline.client_wrapper
    model_metrics['collection_name'] = collection_name
    client_wrapper.save_metrics(model_metrics_dict=model_metrics, log=log)
    pipeline.log.warning('Successfully saved model metrics into the model tracker database.')

# *********************************************************************************************************************************
def get_metrics(pipeline, filter_dict=None, collection_name='model_tracker', log=False):
    """Wrapper for mlmt_client.get_model_metrics(). Retrieves a list of performance metrics dictionaries
    matching a query filter, or by default all performance results for the current model UUID, from all training
    and/or prediction runs.

    Args:
        pipeline (ModelPipeline): The ModelPipeline object managing the training or prediction run.

        filter_dict (dict): An optional dictionary of query criteria. If not specified, the function
        returns all performance results for the current model UUID, from all training and/or prediction runs.

    Returns:
        A list of dicts containing training and prediction performance metrics. Raises MongoQueryException if the query fails.
    """

    if filter_dict is None:
        # Specify model_uuid at least.
        filter_dict = dict(model_uuid = pipeline.params.model_uuid)
    # Temporarily add collection_name key. The model tracker will use this key internally and pop it from the dict.
    filter_dict['collection_name'] = collection_name
    client_wrapper = pipeline.client_wrapper
    # call next(gen) to get the next item.
    gen = client_wrapper.get_metrics_generator(filter_dict=filter_dict, log=log)
    pipeline.log.warning('Successfully created metrics generator.')
    return gen

