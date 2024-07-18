"""Module to interface model pipeline to model tracker service."""

import os
import tempfile
import pandas as pd
import json
import tarfile
import logging

logger = logging.getLogger('ATOM')

from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline import transformations as trans
import atomsci.ddm.utils.file_utils as futils

mlmt_supported = True
try:
    from atomsci.clients import MLMTClient
except (ModuleNotFoundError, ImportError):
    logger.debug("Model tracker client not supported in your environment; will save models in filesystem only.")
    mlmt_supported = False

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
        None if insertion was successful, raises DatastoreInsertionException, MLMTClientInstantiationException
        or MongoInsertionException otherwise
    """
    
    if pipeline is None:
        raise Exception('pipeline cannot be None.')

    if not mlmt_supported:
        logger.error("Model tracker not supported in your environment; can save models in filesystem only.")
        return

    # ModelPipeline.create_model_metadata() should be called before the call to save_model.
    # Get the metadata dictionary from the model pipeline.
    metadata_dict = pipeline.model_metadata
    model_uuid = metadata_dict['model_uuid']
    if model_uuid is None:
        raise ValueError("model_uuid is missing from pipeline metadata.")

    #### Part 1: Save the model tarball in the datastore ####
    model = pipeline.model_wrapper
    # Put tar file in a temporary directory that will automatically be destroyed when we're done
    with tempfile.TemporaryDirectory() as tmp_dir:
        tarball_path = os.path.join(tmp_dir, f"model_{model_uuid}.tar.gz")
        save_model_tarball(pipeline.params.output_dir, tarball_path)

        title = f"{model_uuid} model tarball"
        ds_key = f"model_{model_uuid}_tarball"
        uploaded_results = dsf.upload_file_to_DS(
            bucket=pipeline.params.model_bucket, title=title, description=title, tags=[],
            key_values={'model_uuid' : model_uuid, 'file_category': 'ml_model'}, filepath=tmp_dir,
            filename=tarball_path, dataset_key=ds_key, client=pipeline.ds_client,
            return_metadata=True)
        if uploaded_results is None:
            raise DatastoreInsertionException('Unable to upload title={title} to datastore.'.format(title=title))
    # Get the dataset_oid for actual metadata file stored in datastore.
    model_dataset_oid = uploaded_results['dataset_oid']
    # By adding dataset_oid to the dict, we can immediately find the datastore file asssociated with a model.
    metadata_dict['model_parameters']['model_dataset_oid'] = model_dataset_oid


    #### Part 2: Save the model metadata in the model tracker ####
    mlmt_client = dsf.initialize_model_tracker()
    mlmt_client.save_metadata(collection_name=collection_name,
                                    model_uuid=metadata_dict['model_uuid'],
                                    model_metadata=metadata_dict)
    if log:
        logger.info('Successfully inserted into the database with model_uuid %s.' % model_uuid)

# *********************************************************************************************************************************
def get_full_metadata(filter_dict, collection_name=None):
    """Retrieve relevant full metadata (including training run metrics) of models matching given criteria.

    Args:
        filter_dict (dict): dictionary to filter on

        collection_name (str): Name of collection to search

    Returns:
        A list of matching full model metadata (including training run metrics) dictionaries. Raises MongoQueryException if the query fails.
    """
    if not mlmt_supported:
        logger.error("Model tracker not supported in your environment; can load models from filesystem only.")
        return None

    if filter_dict is None:
        raise ValueError('Parameter filter_dict cannot be None.')
    if collection_name is None:
        raise ValueError('Parameter collection_name cannot be None.')
    mlmt_client = dsf.initialize_model_tracker()

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

    if not mlmt_supported:
        logger.error("Model tracker not supported in your environment; can load models from filesystem only.")
        return None

    mlmt_client = dsf.initialize_model_tracker()

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

    if not mlmt_supported:
        logger.error("Model tracker not supported in your environment; can load models from filesystem only.")
        return None

    mlmt_client = dsf.initialize_model_tracker()

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

    if not mlmt_supported:
        logger.error("Model tracker not supported in your environment; can load models from filesystem only.")
        return None

    mlmt_client = dsf.initialize_model_tracker()

    collections = mlmt_client.collections.get_collection_names().result()
    for col in collections:
        if not col.startswith('old_'):
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
    if not mlmt_supported:
        logger.error("Model tracker not supported in your environment; can load models from filesystem only.")
        return None

    model_meta = get_metadata_by_uuid(uuid)
    response_col = model_meta['training_dataset']['response_cols']
    smiles_col = model_meta['training_dataset']['smiles_col']
    id_col = model_meta['training_dataset']['id_col']
    full_data  = dsf.retrieve_dataset_by_dataset_oid(model_meta['training_dataset']['dataset_oid'])

    # Pull split data and merge into initial dataset
    split_meta = dsf.search_datasets_by_key_value('split_dataset_uuid', model_meta['splitting_parameters']['split_uuid'])
    split_oid  = split_meta['dataset_oid'].values[0]
    split_data = dsf.retrieve_dataset_by_dataset_oid(split_oid)
    split_data['compound_id'] = split_data['cmpd_id']
    split_data = split_data.drop(columns=['cmpd_id'])
    full_data = pd.merge(full_data, split_data, how='inner',
            left_on=[id_col], right_on=['compound_id'])

    train_data = full_data[full_data['subset'] == 'train'][['compound_id',smiles_col,id_col,*response_col]].reset_index(drop=True)
    valid_data = full_data[full_data['subset'] == 'valid'][['compound_id',smiles_col,id_col,*response_col]].reset_index(drop=True)
    test_data  = full_data[full_data['subset'] == 'test'][['compound_id',smiles_col,id_col,*response_col]].reset_index(drop=True)

    return train_data, valid_data, test_data


# *********************************************************************************************************************************
def save_model_tarball(output_dir, model_tarball_path):
    """Save the model parameters, metadata and transformers as a portable gzipped tar archive.

    Args:
        output_dir (str): Output directory from model training

        model_tarball_path (str): Path of tarball file to be created

    Returns:
        None
    """
    with tarfile.open(model_tarball_path, mode='w:gz') as tarball:
        for filename in ['best_model', 'model_metadata.json', 'model_metrics.json']:
            tarball.add(f"{output_dir}/{filename}", arcname=f"./{filename}")
        if os.path.exists(f"{output_dir}/transformers.pkl"):
            tarball.add(f"{output_dir}/transformers.pkl", arcname='./transformers.pkl')
    logger.info(f"Wrote model tarball to {model_tarball_path}")


# *********************************************************************************************************************************
def extract_datastore_model_tarball(model_uuid, model_bucket, output_dir, model_dir):
    """Load a model tarball saved in the datastore and check the format. If it is a new style tarball (containing
    the model metadata and transformers along with the model state), unpack it into output_dir. Otherwise
    it contains the model state only; unpack it into model_dir.

    Args:
        model_uuid (str): UUID of model to be retrieved

        model_bucket (str): Datastore bucket containing model tarball file

        output_dir (str): Output directory to unpack tarball into if it's in the new format

        model_dir (str): Output directory to unpack tarball into if it's in the old format

    Returns:
        extract_dir (str): The directory (output_dir or model_dir) the tarball was extracted into.
    """

    ds_client = dsf.config_client()
    model_dataset_key = 'model_%s_tarball' % model_uuid

    # Download the tarball to a temporary file so we can analyze it and extract its contents. Unfortunately the tarfile
    # module prevents us from using the datastore client stream directly because it requires that the stream be seekable.
    tarfile_fp = tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False)
    tarfile_path = tarfile_fp.name
    with ds_client.open_bucket_dataset(model_bucket, model_dataset_key, mode='b') as dstore_fp:
        for chunk in dstore_fp:
            if chunk:
                tarfile_fp.write(chunk)
    tarfile_fp.close()

    # Look at the tarball contents and figure out which format it's in. If it already has the metadata.json
    # and transformers, extract it into output_dir; otherwise into model_dir.
    with open(tarfile_path, 'rb') as tarfile_fp:
        with tarfile.open(fileobj=tarfile_fp, mode='r:gz') as tfile:
            tar_contents = tfile.getnames()
            if './model_metadata.json' in tar_contents:
                extract_dir = output_dir
            else:
                extract_dir = model_dir
            os.makedirs(extract_dir, exist_ok=True)
            futils.safe_extract(tfile, path=extract_dir)

    os.remove(tarfile_path)
    logger.info(f"Extracted model tarball contents to {extract_dir}")
    return extract_dir

# *********************************************************************************************************************************
def export_model(model_uuid, collection, model_dir, alt_bucket='CRADA'):
    """Export the metadata (parameters) and other files needed to recreate a model
    from the model tracker database to a gzipped tar archive.

    Args:
        model_uuid (str): Model unique identifier

        collection (str): Name of the collection holding the model in the database.

        model_dir (str): Path to directory where the model metadata and parameter files will be written. The directory will
        be created if it doesn't already exist. Subsequently, the directory contents will be packed into a gzipped tar archive
        named model_dir.tar.gz.

        alt_bucket (str): Alternate datastore bucket to search for model tarball and transformer objects.

    Returns:
        none
    """
    if not mlmt_supported:
        logger.info("Model tracker not supported in your environment; can load models from filesystem only.")
        return

    ds_client = dsf.config_client()
    metadata_dict = get_metadata_by_uuid(model_uuid, collection_name=collection)

    output_dir = model_dir
    model_dir = f"{output_dir}/best_model"

    # Convert metadata if it's in the old camelcase format (shouldn't exist anymore)
    if 'ModelMetadata' in metadata_dict:
        # Convert old style metadata
        metadata_dict = convert_metadata(metadata_dict)

    if 'model_parameters' in metadata_dict:
        model_parameters = metadata_dict['model_parameters']
    else:
        raise Exception("Bad metadata for model UUID %s" % model_uuid)


    model_params = parse.wrapper(metadata_dict)

    # Override selected model training parameters

    # Check that buckets where model tarball and transformers were saved still exist. If not, try alt_bucket.
    trans_bucket_differs = (model_params.transformer_bucket != model_params.model_bucket)
    model_bucket_meta = ds_client.ds_buckets.get_buckets(buckets=[model_params.model_bucket]).result()
    if len(model_bucket_meta) == 0:
        model_params.model_bucket = alt_bucket
    if trans_bucket_differs:
        trans_bucket_meta = ds_client.ds_buckets.get_buckets(buckets=[model_params.transformer_bucket]).result()
        if len(trans_bucket_meta) == 0:
            model_params.transformer_bucket = alt_bucket
    else:
        if len(model_bucket_meta) == 0:
            model_params.transformer_bucket = alt_bucket

    # Get the tarball containing the saved model from the datastore, and extract it into output_dir or model_dir.
    extract_dir = extract_datastore_model_tarball(model_uuid, model_params.model_bucket, output_dir, model_dir)

    if extract_dir == model_dir:
        # Download the transformers pickle file if there is one
        if trans.transformers_needed(model_params):
            try:
                if model_params.transformer_key is None:
                    transformer_key = 'transformers_%s.pkl' % model_uuid
                else:
                    transformer_key = model_params.transformer_key
                trans_fp = ds_client.open_bucket_dataset(model_params.transformer_bucket, transformer_key, mode='b')
                trans_data = trans_fp.read()
                trans_fp.close()
                trans_path = "%s/transformers.pkl" % output_dir
                trans_out = open(trans_path, mode='wb')
                trans_out.write(trans_data)
                trans_out.close()
                del model_parameters['transformer_oid']
                model_parameters['transformer_key'] = 'transformers.pkl'
        
            except:
                logger.info("Transformers expected but not found in datastore in bucket %s with key\n%s" 
                             % (model_params.transformer_bucket, transformer_key))
                raise
    
        # Save the metadata params
        model_parameters['save_results'] = False
        meta_path = f"{output_dir}/model_metadata.json"
        with open(meta_path, 'w') as meta_out:
            json.dump(metadata_dict, meta_out, indent=4)
    
        # Save the metrics to model_metrics.json
        if 'training_metrics' in metadata_dict:
            model_metrics = metadata_dict['training_metrics']
            metrics_path = f"{output_dir}/model_metrics.json"
            with open(metrics_path, 'w') as metrics_out:
                json.dump(model_metrics, metrics_out, indent=4)
        else:
            logger.info(f"No metrics saved for model {model_uuid}")

    # Create a new tarball containing both the metadata and the parameters from the retrieved model tarball
    new_tarpath = "%s.tar.gz" % output_dir
    tarball = tarfile.open(new_tarpath, mode='w:gz')
    tarball.add(output_dir, arcname='.')
    tarball.close()
    logger.info("Wrote model files to %s" % new_tarpath)


# *********************************************************************************************************************************
def convert_metadata(old_metadata):
    """Convert model metadata from old format (with camel-case parameter group names) to new format.

    Args:
        old_metadata (dict): Model metadata in old format

    Returns:
        new_metadata (dict): Model metadata in new format
    """

    model_metadata = old_metadata['ModelMetadata']
    model_parameters = model_metadata['ModelParameters']
    training_dataset = model_metadata['TrainingDataset'].copy()
    new_metadata = {
        "model_uuid": old_metadata['model_uuid'],
        "time_built": old_metadata['time_built'],
        "training_dataset": training_dataset,
        "training_metrics": []
    }

    map_keys = [
        ("external_export_parameters", "ExternalExportParameters"),
        ("dataset_metadata", "DatasetMetadata"),
    ]

    for (nkey, okey) in map_keys:
        value = training_dataset.pop(okey, None)
        if value is not None:
            training_dataset[nkey] = value

    map_keys = [
        ("model_parameters", 'ModelParameters'),
        ("ecfp_specific", 'ECFPSpecific'),
        ("rf_specific", 'RFSpecific'),
        ("autoencoder_specific", 'AutoencoderSpecific'),
        ("descriptor_specific", 'DescriptorSpecific'),
        ("nn_specific", "NNSpecific"),
        ("xgb_specific", "xgbSpecific"),
        ("umap_specific", "UmapSpecific"),

    ]
    for (nkey, okey) in map_keys:
        value = model_metadata.get(okey)
        if value is not None:
            new_metadata[nkey] = value

    # Get rid of useless extra level in split params
    split_params = model_metadata.get('SplittingParameters')
    if split_params is not None:
        splitting = split_params.get('Splitting')
        if splitting is not None:
            new_metadata['splitting_parameters'] = splitting

    return new_metadata
