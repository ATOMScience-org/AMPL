import glob
import os
import shutil
import inspect
import pandas as pd
import requests

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import atomsci.ddm.pipeline.parameter_parser as parse

import atomsci.ddm.pipeline.model_pipeline as MP
import atomsci.ddm.pipeline.featurization as feat
import atomsci.ddm.pipeline.model_datasets as model_dataset
import atomsci.ddm.utils.curate_data as curate_data
import atomsci.ddm.utils.struct_utils as struct_utils

import logging
logging.basicConfig(format='%(asctime)-15s %(message)s')
log = logging.getLogger('ATOM')

def clean():
    split_files = glob.glob('./*_train_valid_test_*')
    for f in split_files:
        os.remove(f)

    if os.path.exists('pytest'):
        shutil.rmtree('pytest')

def datastore_status():
    """Returns True if the datastore is down, False otherwise. Will figure out how to do this programmatically"""
    try:
        a = datastore_objects()
    except Exception as e:
        log.warning("datastore connection failed: " + str(e))
        return True
    else:
        return False

def delaney_objects(y=["measured log solubility in mols per litre"], featurizer="ecfp", split_strategy="train_valid_test", splitter = "random", split_uuid=None):
    delaney_inp_file = currentdir + '/config_delaney.json'
    inp_params = parse.wrapper(delaney_inp_file)
    inp_params.response_cols = y
    inp_params.featurizer = featurizer
    inp_params.split_strategy = split_strategy
    inp_params.splitter = splitter
    if split_uuid is not None:
        inp_params.previously_split = True
        inp_params.split_uuid = split_uuid
    featurization = feat.create_featurization(inp_params)
    mdl = model_dataset.create_model_dataset(inp_params, featurization, ds_client = None)
    delaney_df = mdl.load_full_dataset()
    mdl.get_featurized_data()
    mdl.split_dataset()
    return inp_params, mdl, delaney_df

def delaney_pipeline(y=["measured log solubility in mols per litre"], featurizer="ecfp", split_strategy="train_valid_test", splitter = "random"):
    delaney_inp_file = currentdir + '/config_delaney.json'
    inp_params = parse.wrapper(delaney_inp_file)
    inp_params.response_cols = y
    inp_params.featurizer = featurizer
    inp_params.split_strategy = split_strategy
    inp_params.splitter = splitter
    mp = MP.ModelPipeline(inp_params)
    return mp

def datastore_objects(y = ["PIC50"]):
    params_from_ds = parse.wrapper(currentdir + '/config_datastore_dset_cav12.json')
    params_from_ds.response_cols = y
    featurization = feat.create_featurization(params_from_ds)
    data = model_dataset.create_model_dataset(params_from_ds, featurization)
    dset_df = data.load_full_dataset()
    data.get_featurized_data()
    data.split_dataset()
    return params_from_ds, data, dset_df

# TODO (ksm): The object referenced by this config file is no longer in the datastore. I replaced
# calls to this function with the datastore_objects function above.
def uncurated_objects(y = ["VALUE_NUM"]):
    params_from_ds = parse.wrapper(currentdir + '/config_uncurated_bp.json')
    params_from_ds.response_cols = y
    featurization = feat.create_featurization(params_from_ds)
    data = model_dataset.create_model_dataset(params_from_ds, featurization)
    uncurated_df = data.load_full_dataset()
    return params_from_ds, data, uncurated_df

def moe_descriptors(datastore = False):
    if datastore == True:
        params_ds = parse.wrapper(currentdir + "/config_MAOA_moe_descriptors_ds.json")
    else:
        params_file = parse.wrapper(currentdir + "/config_MAOA_moe_descriptors.json")
#         if not os.path.isfile(params_file.dataset_key):
#             os.makedirs('pytest/config_MAOA_moe_descriptors/moe_descriptors', exist_ok=True)
#             copyfile(params_ds.dataset_key, params_file.dataset_key)
    if datastore == True:
        params_desc = params_ds
    else: 
        params_desc = params_file
    featurization = feat.create_featurization(params_desc)
    dataset_obj_for_desc = model_dataset.create_model_dataset(params_desc, featurization, ds_client = None)
    df = dataset_obj_for_desc.load_full_dataset()
    return params_desc, dataset_obj_for_desc, df


def curate_delaney():
    """Curate dataset for model fitting"""
    if (not os.path.isfile('delaney-processed_curated.csv') and
            not os.path.isfile('delaney-processed_curated_fit.csv') and
            not os.path.isfile('delaney-processed_curated_external.csv')):
        raw_df = pd.read_csv('delaney-processed.csv')

        # Generate smiles, inchi
        raw_df['rdkit_smiles'] = raw_df['smiles'].apply(curate_data.base_smiles_from_smiles)
        raw_df['inchi_key'] = raw_df['smiles'].apply(struct_utils.smiles_to_inchi_key)

        # Check for duplicate compounds based on SMILES string
        # Average the response value for duplicates
        # Remove compounds where response value variation is above the threshold
        # tolerance=% of individual respsonse value is allowed to different from the average to be included in averaging.
        # max_std = maximum allowed standard deviation for computed average response value
        tolerance = 10  # percentage
        column = 'measured log solubility in mols per litre'
        list_bad_duplicates = 'Yes'
        data = raw_df
        max_std = 100000  # esentially turned off in this example
        data['compound_id'] = data['inchi_key']
        curated_df = curate_data.average_and_remove_duplicates(
            column, tolerance, list_bad_duplicates, data, max_std, compound_id='compound_id', smiles_col='rdkit_smiles')

        # Check distribution of response values
        assert (curated_df.shape[0] == 1116), 'Error: Incorrect number of compounds'

        curated_df.to_csv('delaney-processed_curated.csv')

        # Create second test set by reproducible index for prediction
        curated_df.tail(999).to_csv('delaney-processed_curated_fit.csv')
        curated_df.head(117).to_csv('delaney-processed_curated_external.csv')

    assert (os.path.isfile('delaney-processed_curated.csv'))
    assert (os.path.isfile('delaney-processed_curated_fit.csv'))
    assert (os.path.isfile('delaney-processed_curated_external.csv'))


def download_delaney():
    """Separate download function so that download can be run separately if there is no internet."""
    if (not os.path.isfile('delaney-processed.csv')):
        download_save(
            'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
            'delaney-processed.csv')

    assert (os.path.isfile('delaney-processed.csv'))


def download_save(url, file_name, verify=True):
    """Download dataset

    Arguments:
        url: URL
        file_name: File name
        verify: Verify SSL certificate
    """

    # Skip if file already exists
    if (not (os.path.isfile(file_name) and os.path.getsize(file_name) > 0)):
        assert(url)
        r = requests.get(url, verify=verify)

        assert (r.status_code == 200), 'Error: Could not download dataset'

        with open(file_name, 'wb') as f:
            f.write(r.content)

        assert (os.path.isfile(file_name) and os.path.getsize(file_name) > 0), 'Error: dataset file not created'


clean()
download_delaney()
curate_delaney()
