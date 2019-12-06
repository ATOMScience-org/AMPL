"""
Module to interface model pipeline to model tracker service.
"""

import os
import subprocess
import tempfile
import sys
import pandas as pd

from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.clients import MLMTClient

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
    metadata_dict['model_parameters']['model_dataset_oid'] = model_dataset_oid


    #### Part 2: Save the model metadata ####
    mlmt_client = MLMTClient()
    mlmt_client.save_metadata(collection_name=collection_name,
                                    model_uuid=metadata_dict['model_uuid'],
                                    model_metadata=metadata_dict)
    if log:
        print('Successfully inserted into the database with model_uuid %s.' % model_uuid)

# *********************************************************************************************************************************
def get_full_metadata(filter_dict, collection_name=None):
    """Retrieve relevant full metadata (including training run metrics) of models matching given criteria.

    Args:
        filter_dict (dict): dictionary to filter on

        collection_name (str): Name of collection to search

    Returns:
        A list of matching full model metadata (including training run metrics) dictionaries. Raises MongoQueryException if the query fails.
    """
    if filter_dict is None:
        raise ValueError('Parameter filter_dict cannot be None.')
    if collection_name is None:
        raise ValueError('Parameter collection_name cannot be None.')
    mlmt_client = MLMTClient()

    query_params = {
        "match_metadata": filter_dict,
    }

    metadata_list = mlmt_client.model.query_model_metadata(
        collection_name=collection_name,
        query_params=query_params
    ).result()
    return list(metadata_list)

# *********************************************************************************************************************************
def get_metadata_by_uuid(model_uuid, collection_name=None):
    """Retrieve model parameter metadata by model_uuid. The resulting metadata dictionary can
    be passed to parameter_parser.wrapper(); it does not contain performance metrics or
    training dataset metadata.

    Args:
        model_uuid (str): model unique identifier
        collection_name(str): collection to search (optional, searches all collections if not specified)
    Returns:
        Matching metadata dictionary. Raises MongoQueryException if the query fails.
    """

    mlmt_client = MLMTClient()

    if collection_name is None:
        collection_name = get_model_collection_by_uuid(model_uuid, mlmt_client=mlmt_client)

    exclude_fields = [
        "training_metrics",
        "time_built",
        "training_dataset.dataset_metadata"
    ]
    return mlmt_client.get_model(collection_name=collection_name, model_uuid=model_uuid,
                                 exclude_fields=exclude_fields)

# *********************************************************************************************************************************
def get_full_metadata_by_uuid(model_uuid, collection_name=None):
    """Retrieve model parameter metadata for the given model_uuid and collection.
    The returned metadata dictionary will include training run performance metrics and
    training dataset metadata.

    Args:
        model_uuid (str): model unique identifier
        collection_name(str): collection to search (optional, searches all collections if not specified)
    Returns:
        Matching metadata dictionary. Raises MongoQueryException if the query fails.
    """

    mlmt_client = MLMTClient()

    if collection_name is None:
        collection_name = get_model_collection_by_uuid(model_uuid, mlmt_client=mlmt_client)

    return mlmt_client.get_model(collection_name=collection_name, model_uuid=model_uuid)

# *********************************************************************************************************************************
def get_model_collection_by_uuid(model_uuid, mlmt_client=None):
    """Retrieve model collection given a uuid.

    Args:
        model_uuid (str): model uuid

        mlmt_client: Ignored
    Returns:
        Matching collection name
    Raises:
        ValueError if there is no collection containing a model with the given uuid.
    """

    mlmt_client = MLMTClient()

    collections = mlmt_client.collections.get_collection_names().result()
    for col in collections:
        if mlmt_client.count_models(collection_name=col, model_uuid=model_uuid) > 0:
            return col

    raise ValueError('Collection not found for uuid: ' + model_uuid)

# *********************************************************************************************************************************
def get_model_training_data_by_uuid(uuid):
    """Retrieve data used to train, validate, and test a model given the uuid

    Args:
        uuid (str): model uuid
    Returns:
        a tuple of datafraes containint training data, validation data, and test data including the compound ID, RDKIT SMILES, and response value
    """
    model_meta = get_metadata_by_uuid(uuid)
    response_col = model_meta['training_dataset']['response_cols']
    smiles_col = model_meta['training_dataset']['smiles_col']
    full_data  = dsf.retrieve_dataset_by_dataset_oid(model_meta['training_dataset']['dataset_oid'], verbose=False)

    # Pull split data and merge into initial dataset
    split_meta = dsf.search_datasets_by_key_value('split_dataset_uuid', model_meta['splitting_parameters']['Splitting']['split_uuid'])
    split_oid  = split_meta['dataset_oid'].values[0]
    split_data = dsf.retrieve_dataset_by_dataset_oid(split_oid, verbose=False)
    split_data['compound_id'] = split_data['cmpd_id']
    split_data = split_data.drop(columns=['cmpd_id'])
    full_data = pd.merge(full_data, split_data, how='inner', on=['compound_id'])

    train_data = full_data[full_data['subset'] == 'train'][['compound_id',smiles_col,*response_col]].reset_index(drop=True)
    valid_data = full_data[full_data['subset'] == 'valid'][['compound_id',smiles_col,*response_col]].reset_index(drop=True)
    test_data  = full_data[full_data['subset'] == 'test'][['compound_id',smiles_col,*response_col]].reset_index(drop=True)

    return train_data, valid_data, test_data



