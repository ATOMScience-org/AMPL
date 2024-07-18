"""This file contains functions to make it easier to browse and retrieve data from the datastore.
   Intended for general use. Add/modify functions as needed. Created 23Jul18 CHW
"""

# -------------setup section-----------------

import sys
import io
import json
import urllib3
import bravado
import os
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger('ATOM')

import csv
import bz2
import pprint
urllib3.disable_warnings()
import pickle
import tarfile
import tempfile
import getpass

from atomsci.ddm.utils.llnl_utils import is_lc_system
import atomsci.ddm.utils.file_utils as futils

feather_supported = True
try:
    import pyarrow.feather as feather
except (ImportError, AttributeError, ModuleNotFoundError):
    feather_supported = False

clients_supported = True
try:
    from atomsci.clients import DatastoreClient
    from atomsci.clients import DatastoreClientSingleton
    from atomsci.clients import MLMTClient
    from atomsci.clients import MLMTClientSingleton
except (ModuleNotFoundError, ImportError):
    logger.info("atomsci.clients package missing, is currently unsupported for non-ATOM users.\n" +
                "ATOM users should run 'pip install clients --user' to install.")
    clients_supported = False

## You must load your token to get access to the appropriate bucket where data is to be placed (or retrieved)
# refer to documentation on Confluence for how to create your token



#===function definition section==========================================================================

def config_client(
        token=None,
        url='https://twintron-blue.llnl.gov/atom/datastore/api/v1.0/swagger.json',
        new_instance=False):
    """Configures client to access datastore service.

    Args:
        token (str): Path to file containing token for accessing datastore. Defaults to
        /usr/local/data/ds_token.txt on non-LC systems, or to $HOME/data/ds_token.txt on LC systems.

        url (str): URL for datastore REST service.

        new_instance (bool): True to force creation of a new client object. By default, a shared
        singleton object is returned.

    Returns:
        returns configured client
    """

    if not clients_supported:
        raise Exception("Datastore client not supported in current environment.")
        
    if token is None:
        # Default token path depends on whether you're on LC or another LLNL system
        if is_lc_system():
            token = os.path.join(os.environ['HOME'], 'data', 'ds_token.txt')
        else:
            token = '/usr/local/data/ds_token.txt'
    token_str = None
    if 'DATASTORE_API_TOKEN' in os.environ:
        token_str = os.environ['DATASTORE_API_TOKEN']
    else:
        if os.path.exists(token):
            with open(token,'r') as f:
                token_str = f.readline().strip()
            os.environ['DATASTORE_API_TOKEN'] = token_str

    if new_instance:
        client = DatastoreClient(default_url=url,
                                 default_api_token=token_str)
    else:
        client = DatastoreClientSingleton(default_url=url,
                                          default_api_token=token_str)

    if not client.api_token:
        if not token_str:
            logger.error("token file not found: {}".format(token))
            
        logger.error("and none of {} token env vars set".format(",".join(
            DatastoreClient.api_token_env_str)))

    return client


#--------------------------------------------------------------------------------------------------------

def initialize_model_tracker(new_instance=False): 
    """Create or obtain a client object for the model tracker service..

    Returns:
        mlmt_client (MLMTClientSingleton): The client object for the model tracker service.
    """
    if not clients_supported:
        raise Exception("Model tracker client not supported in current environment.")

    if 'MLMT_REST_API_URL' not in os.environ:
        os.environ['MLMT_REST_API_URL'] = 'https://twintron-blue.llnl.gov/atom/mlmt/api/v1.0/swagger.json'

    # MLMT service uses same API token as datastore. Make sure it gets set in the environment.
    ds_client = config_client()

    if new_instance:
        mlmt_client = MLMTClient()
    else:
        mlmt_client = MLMTClientSingleton()

    return mlmt_client

#--------------------------------------------------------------------------------------------------------

def retrieve_bucket_names(client=None):
    """Retrieve a list of the bucket names in datastore

    Args:
        client (optional): set client if not using the default

    Returns:
        (list): list of bucket names that exist in the datastore which user has access to
    """

    if client is None:
        client = config_client()

    buckets = client.ds_buckets.get_buckets().result()
    buckets = pd.DataFrame(buckets)
    buckets = list(buckets['bucket_name'])
    return buckets


#------------------------------------------------------------------------------------------------------
def retrieve_keys(bucket='all', client=None, sort=True):
    """Get a list of keys in bucket(s) specified.

    Args:
        bucket (str, optional): 'all' by default. Specify bucket (as a str or list) to limit search

        client (optional): set client if not using the default

        sort (bool, optional): if 'True' (default), sort the keys alphabetically

    Returns:
        (list): returns a list of keys in bucket(s) specified
    """

    if client is None:
        client = config_client()

    if bucket == 'all':
        logger.info('retrieving keys for all buckets...')
        keys = client.ds_metadef.get_metadata_keys().result()
    else:
        if type(bucket) == str:
            bucket = [bucket]

        # check if requested buckets are valid bucket names
        all_buckets = retrieve_bucket_names(client)
        valid_buckets = [i for i in all_buckets if i in bucket] #compares the list of 'valid' buckets with the requested list

        if (len(valid_buckets) > 0) and (len(valid_buckets) < len(bucket)):
            print("Not all buckets requested all valid buckets. Keys will be retrieved for the following buckets:")
            bucket = valid_buckets
            print(bucket)
        if len(valid_buckets) == 0:
            print("Requested bucket(s) are not valid.")
            return
        keys = client.ds_metadef.get_metadata_keys(bucket_names = bucket).result()

        if sort:
            keys = sorted(keys, key = str.lower)

    return keys

#------------------------------------------------------------------------------------------------------
def key_exists(key, bucket='all', client=None):
    """Check if key exists in bucket(s) specified.

    Args:
        key (str): the key of interest

        bucket (str or list, optional): 'all' by default. Specify bucket (as a str or list) to limit search

        client (optional): set client if not using the default

    Returns:
        (bool): Returns True if key exists in bucket(s) specified
    """

    if client is None:
        client = config_client()

    if type(key) != str:
        raise ValueError("'key' must be a string")

    if bucket == 'all':
        # generate list of valid keys for all buckets
        keys = client.ds_metadef.get_metadata_keys().result()
    else:
        if type(bucket) == str:
            bucket = [bucket]

         # check if requested buckets are valid bucket names
        all_buckets = retrieve_bucket_names(client)
        valid_buckets = [i for i in all_buckets if i in bucket] #compares the list of 'valid' buckets with the requested list

        if (len(valid_buckets) > 0) and (len(valid_buckets) < len(bucket)):
            print("Not all buckets requested all valid buckets. Keys will be retrieved for the following buckets:")
            bucket = valid_buckets
            print(bucket)
        if len(valid_buckets) == 0:
            raise ValueError("Requested bucket(s) are not valid.")

        # generate list of valid keys for bucket(s) specified
        keys = client.ds_metadef.get_metadata_keys(bucket_names = bucket).result()

    #check if key specified is in 'valid key' list
    return key in keys


#------------------------------------------------------------------------------------------------------
def retrieve_values_for_key(key, bucket='all', client=None):
    """Get a list of values associated with a specified key.

    Args:
        key (str): the key of interest

        bucket (str or list, optional): 'all' by default. Specify bucket (as a str or list) to limit search

        client (optional): set client if not using the default

    Returns:
        (list): Returns a list of values (str) associated with a specified key
    """

    if client is None:
        client = config_client()

    if type(key) != str:
        raise ValueError('key must be a string')

    if bucket == 'all':
        # evaluate if key is valid
        all_keys = retrieve_keys()
        if key not in all_keys:
            raise ValueError('specified key does not exist')

        values = client.ds_metadef.get_metadata_key_values(key=key).result()
        value_type = values['value_types']
        values = values['values']

    else:
        if type(bucket) == str:
            bucket = [bucket]

        # evaluate if bucket name is valid
        for i in bucket:
            bucket_name = i
            all_buckets = retrieve_bucket_names(client)
            if bucket_name not in all_buckets:
                raise ValueError('bucket does not exist')

        # evaluate if key is valid
        all_keys = retrieve_keys(bucket=bucket)
        if key not in all_keys:
            raise ValueError('specified key does not exist in bucket(s) specified')

        values = client.ds_metadef.get_metadata_key_values(key=key, bucket_names = bucket).result()
        value_type = values['value_types']
        values = values['values']

    if value_type == ['str']:
        values = sorted(values, key = str.lower)

    if value_type == ['int']:
        values = np.sort(values)

    return values


#------------------------------------------------------------------------------------------------------
def dataset_key_exists(dataset_key, bucket, client=None):
    """Returns a boolean indicating whether the given dataset_key is already present in the bucket specified.

    Args:
        dataset_key (str): the dataset_key for the dataset you want (unique in each bucket)

        bucket (str): the bucket the dataset you want resides in

        client (optional): set client if not using the default

    Returns:
        (bool): returns 'True' if dataset_key is present in bucket specified
    """

    if client is None:
        client = config_client()

    # check that bucket exists
    all_buckets = retrieve_bucket_names(client)
    if bucket not in all_buckets:
        raise ValueError('bucket does not exist')

    # check that dataset_key exists in bucket
    all_dataset_keys = client.ds_datasets.get_dataset_distinct_dataset_keys(bucket_name=bucket).result()
    return (dataset_key in all_dataset_keys)


#------------------------------------------------------------------------------------------------------
def retrieve_dataset_by_datasetkey(dataset_key, bucket, client=None, return_metadata=False, nrows=None, print_metadata=False, sep=False, index_col=None, tarpath=".", **kwargs):
    """Retrieves the dataset and returns as a pandas dataframe (or other format as needed depending on file type).

    Args:
        dataset_key (str): the dataset_key for the dataset you want (unique in each bucket)

        bucket (str): the bucket the dataset you want resides in

        client (optional): set client if not using the default

        return_metadata (bool, optional): if set to True, return a dictionary of the metadata INSTEAD of a dataframe of the data

        nrows (num, optional): used to limit the number of rows returned

        print_metadata (bool, optional): if set to True, displays the document metadata/properties

        sep (str, optional): separator used for csv file

        tarpath (str, optional): path to use for tarball files

        index_col (int, optional): For csv files, column to use as the row labels of the DataFrame

    Returns:
        (DataFrame, OrderedDict, str, dict): filetype determines what type of object is returned.
        xls and xlsx files returns an OrderedDict.
        tarball (gz and tgz) files returns the location of the files as a string
        csv returns a DataFrame

        optionally, return a dictionary of the metadata only if 'return_metadata' is set to TRUE.
    """

    if client is None:
        client = config_client()

    # check that bucket exists
    all_buckets = retrieve_bucket_names(client)
    if bucket not in all_buckets:
        raise ValueError('bucket does not exist')

    # check that dataset_key exists in bucket
    # JEA comment:
    # this is not going to scale well to millions of keys
    # and an exception will already be thrown if the dataset key is not found
    # I think this should be removed
    #all_dataset_keys = client.ds_datasets.get_dataset_distinct_dataset_keys(bucket_name=bucket).result()
    #if not dataset_key in all_dataset_keys:
    #    raise ValueError('dataset_key {0} does not exist in bucket {1}'.format(dataset_key, bucket))

    try :
        all_metadata = client.ds_datasets.get_bucket_dataset (bucket_name=bucket, dataset_key=dataset_key).result()
    except bravado.exception.HTTPNotFound:
        return None

    file_type = all_metadata['distribution']['dataType']
    if print_metadata:
        pprint.pprint(all_metadata)

    if return_metadata:
        return all_metadata

    if file_type == 'bz2':
        # bz2.  Read bz2 file in binary mode, then decompress to text. Return as dataframe.
        fp = client.open_bucket_dataset (bucket, dataset_key, mode='b')
        fp = bz2.open (fp, mode='rt')
        dict_reader = csv.DictReader (fp)
        table = []
        i_row = 1
        for row in dict_reader:
            table.append(row)
            if nrows is not None:
                if i_row >= nrows:
                    #i_row += 1
                    break
            i_row += 1

        dataset = pd.DataFrame(table)

    elif file_type == 'pkl':
        # pickle file. Open in binary.
        fp = client.open_bucket_dataset (bucket, dataset_key, mode='b')
        dataset = pickle.load(fp)
        if type(dataset)==bytes:
            dataset = pickle.loads(dataset)
        fp.close()

    elif file_type == 'csv':
        # csv file.  Open in text mode for csv reader. Return as dataframe.
        fp = client.open_bucket_dataset (bucket, dataset_key, mode='rt')
        if not sep:
            dataset = pd.read_csv(fp,nrows=nrows, index_col=index_col, **kwargs)
        else:
            dataset = pd.read_csv(fp, nrows=nrows, sep=sep, index_col=index_col, **kwargs)\

    elif file_type == 'feather':
        # feather file. Return as dataframe.
        if not feather_supported:
            raise ValueError("feather-format not installed in your current environment")
        fp = client.open_bucket_dataset (bucket, dataset_key, mode='b')
        # Have to save the feather file to disk first, because feather.read_dataframe needs to seek
        tmp_fd, tmp_path = tempfile.mkstemp()
        tmp_fp = os.fdopen(tmp_fd, mode='wb')
        while True:
            data = fp.read()
            if len(data) == 0:
                break
            logger.debug("Read %d bytes of data from datastore" % len(data))
            tmp_fp.write(data)
        logger.debug("Wrote data to %s" % tmp_path)
        tmp_fp.close()
        fp.close()
        logger.debug("Reading data into data frame")
        dataset = feather.read_dataframe(tmp_path)
        logger.debug("Done")
        os.unlink(tmp_path)

    elif file_type == 'xls' or file_type == 'xlsx':
        # xls or xlsx file. Return as a ordered dictionary
        fp = client.open_bucket_dataset (bucket, dataset_key, mode='b')
        dataset = pd.read_excel(fp, sheet_name=None)
        num_sheets = len(dataset)
        sheet_names = dataset.keys()
        print('Excel workbook has %s sheets' %(num_sheets), 'Sheet names = ', sheet_names)
        print('tip: use OrderedDict.get(sheet_name) to extract a specific sheet')

    elif file_type == 'gz' or file_type == 'tgz':
        # tar.gz (tarball) file. Extract to path specified and return path.
        fp = client.open_bucket_dataset (bucket, dataset_key, mode='b')
        with tarfile.open(fileobj=fp, mode='r:gz') as tar:
            futils.safe_extract(tar, path=tarpath)

        #get new folder name and return full path
        extracted_dir = all_metadata['distribution'].get('filename')
        extracted_dir = extracted_dir.split(".")[0]
        # TODO: This is misleading; the original filename is not necessarily preserved in the tar file.
        # TODO: Just return tarpath.
        dataset = os.path.join(tarpath, extracted_dir)

    else:
        raise ValueError (dataset_key, 'file type not recognized\n',
                          "all_metadata['distribution']['dataType']:", all_metadata['distribution']['dataType'])


    return dataset

#------------------------------------------------------------------------------------------------------
def retrieve_dataset_by_dataset_oid(dataset_oid, client=None, return_metadata=False, nrows=None, print_metadata=False, sep=False, index_col=None, tarpath="."):
    """retrieves the dataset and returns as a pandas dataframe (or other format as needed depending on file type).

    Args:
       dataset_oid (str): unique identifier for the dataset you want

       client (optional): set client if not using the default

       return_metadata (bool, optional): if set to True, return a dictionary of the metadata INSTEAD of a dataframe of the data

       nrows (num, optional): used to limit the number of rows returned

       print_metadata (bool, optional): if set to True, displays the document metadata/properties

       sep (str, optional): separator used for csv file

       tarpath (str, optional): path to use for tarball files

       index_col (int, optional): For csv files, column to use as the row labels of the DataFrame

    Returns:
       (DataFrame, OrderedDict, str, dict): filetype determines what type of object is returned.
       xls and xlsx files returns an OrderedDict.
       tarball (gz and tgz) files returns the location of the files as a string
       csv returns a DataFrame

       optionally, return a dictionary of the metadata only if 'return_metadata' is set to TRUE.
    """

    print("")
    print('caution: dataset_oid is version specific. Newer versions of this file might be available.')
    print("")


    if client is None:
        client = config_client()

    all_metadata = client.ds_datasets.get_dataset(dataset_oid = dataset_oid).result()


    if print_metadata:
        pprint.pprint(all_metadata)

    if return_metadata:
        return all_metadata

    file_type = all_metadata['distribution']['dataType']

    if file_type == 'bz2':
        # bz2.  Read bz2 file in binary mode, then decompress to text.
        fp = client.open_dataset (dataset_oid, mode='b')
        fp = bz2.open (fp, mode='rt')
        dict_reader = csv.DictReader (fp)
        table = []
        i_row = 1
        for row in dict_reader:
            table.append(row)
            if nrows is not None:
                if i_row >= nrows:
                    #i_row += 1
                    break
            i_row += 1

        dataset = pd.DataFrame(table)

    elif file_type == 'pkl':
        # pickle file. Open in binary
        fp = client.open_dataset (dataset_oid, mode='b')
        dataset = pickle.load(fp)
        if type(dataset)==bytes:
            dataset = pickle.loads(dataset)
        fp.close()

    elif file_type == 'csv':
        # csv file.  Open in text mode for csv reader.
        fp = client.open_dataset (dataset_oid, mode='t')
        if not sep:
            dataset = pd.read_csv(fp, nrows=nrows, index_col=index_col)
        else:
            dataset = pd.read_csv(fp, nrows=nrows, sep=sep, index_col=index_col)

    elif file_type == 'xls' or file_type == 'xlsx':
        # xls or xlsx file. Return as a ordered dictionary
        fp = client.open_dataset (dataset_oid, mode='b')
        dataset = pd.read_excel(fp, sheet_name=None)
        num_sheets = len(dataset)
        sheet_names = dataset.keys()
        print('Excel workbook has %s sheets' %(num_sheets), 'Sheet names = ', sheet_names)
        print('tip: use OrderedDict.get(sheet_name) to extract a specific sheet')

    elif file_type == 'gz' or file_type == 'tgz':
        # tar.gz (tarball) file. Extract to path specified and return path.
        fp = client.open_dataset (dataset_oid, mode='b')
        with tarfile.open(fileobj=fp, mode='r:gz') as tar:
            futils.safe_extract(tar, path=tarpath)

        #get new folder name and return full path
        extracted_dir = all_metadata['distribution'].get('filename')
        extracted_dir = extracted_dir.split(".")[0]
        dataset = os.path.join(tarpath, extracted_dir)


    else:
        raise ValueError ('file type not recognized \n',
                          "all_metadata['distribution']['dataType']:", all_metadata['distribution']['dataType'])

    return dataset

#------------------------------------------------------------------------------------------------------

def search_datasets_by_key_value(key, value, client=None, operator='in', bucket='all', display_all_columns=False):
    """Find datasets by key:value pairs and returns a DataFrame of datasets and associated properties.

    Args:
        key (str): the key of interest

        value (str): the value of interest

        client (optional): set client if not using the default

        operator (str, optional): 'in' by default, but can be changed to any of the following:
                       =, !=, <, <=, >, >=, all, in, not in

        bucket (str or list, optional): 'all' by default. Specify bucket (as a str or list) to limit search

        display_all_columns (bool, optional): If 'False' (default), then show only a selected subset of the columns

    Returns:
        (DataFrame): summary table of the files and relevant metadata matching the criteria specified
    """

    if client is None:
        client = config_client()

    if type(key) != str:
        raise ValueError('key must be a string')

    if type(value) != list:
        value = [value]
    metadata = json.dumps([ {'key': key, 'value': value, 'operator': operator} ])

    if bucket == 'all':
        # evaluate if key is valid
        all_keys = retrieve_keys()
        if key not in all_keys:
            raise ValueError('specified key does not exist')

        datasets = client.ds_datasets.get_datasets(metadata=metadata).result()
    else:
        if type(bucket) != list:
            bucket = [bucket]

        # evaluate if bucket name is valid
        all_buckets = retrieve_bucket_names(client)
        for i in bucket:
            bucket_name = i
            if bucket_name not in all_buckets:
                raise ValueError('bucket does not exist')

        # evaluate if key is valid
        all_keys = retrieve_keys(bucket=bucket)
        if key not in all_keys:
            raise ValueError('specified key does not exist in bucket(s) specified')


        datasets = client.ds_datasets.get_datasets(metadata=metadata, bucket_names=bucket).result()

    datasets = pd.DataFrame(datasets)

    if len(datasets) == 0:
        print('No datasets found matching criteria specified',key,value)
    else:
        if not display_all_columns:
            col = ['bucket_name', 'title', 'dataset_oid', 'dataset_key', 'description',
                   'metadata', 'tags', 'user_perm', 'active', 'versions']
            datasets = datasets[col]

    return datasets


#-------------------------------------------------------------------------------------------------------
   # extracted this function (with small modifications) from join.ipynb
def retrieve_columns_from_dataset (bucket, dataset_key, client=None, max_rows=0, column_names='', return_names=False):

    """Retrieve column(s) from csv file (may be bz2 compressed) in datastore.
       'NA' returned if column not in file (as well as warning message).

    Args:
        return_names (bool): If true, just return column headers from file

        max_rows (int): default=0 which will return all rows

        client (optional): set client if not using the default

    Returns:
       (dict): dictionary corresponding to selected columns

    """
    if client is None:
        client = config_client()

    # Check column_names input.
    if not isinstance (column_names, list):
        if not isinstance (column_names, str):
            raise TypeError ('get_columns_csv: Second argument should be column name or list of column names', file=sys.stderr)
            sys.exit (1)
        else:
            column_names = [column_names]


    dataset_result \
            = client.ds_datasets \
                    .get_bucket_dataset (bucket_name=bucket,
                                         dataset_key=dataset_key).result ()
    dataset_oid = dataset_result.get ('dataset_oid')
    if dataset_result['distribution']['dataType'] == 'bz2':

        # bz2.  Read bz2 file in binary mode, then decompress to text.
        fp = client.open_dataset (dataset_oid, mode='b')
        fp = bz2.open (fp, mode='rt')

    elif dataset_result['distribution']['dataType'] == 'csv':

        # csv file.  Open in text mode for csv reader.
        fp = client.open_dataset (dataset_oid, mode='t')
    else:
        raise ValueError (dataset_key, 'does not appear to be either a csv or bz2 file\n',
                          "dataset_result['distribution']['dataType']:", dataset_result['distribution']['dataType'])

    dict_reader = csv.DictReader (fp)

    # Set up dict to be returned.
    selected_columns = {}
    for column_name in column_names:
        selected_columns[column_name] = []

    # Check which columns in this file.
    header_names = dict_reader.fieldnames

    if return_names:
        return header_names


    column_names_valid_b = []
    for column_name in column_names:
        column_name_valid_b = column_name in header_names
        column_names_valid_b.append (column_name_valid_b)
        if not column_name_valid_b:
            print ('Note: column', column_name, 'not in', os.path.split (dataset_key)[1], file=sys.stderr)


    print ('Reading ' + os.path.split (dataset_key)[1] + '... ')
    i_row = 1
    for row in dict_reader:
        for i, column_name in enumerate (column_names):
            if column_names_valid_b[i]:
                selected_columns[column_name].append (row[column_name])
            else:
                selected_columns[column_name].append ('NA')

        if i_row % 1000 == 0:
            print (i_row, 'rows                    ', end='\r', flush=True)

        if max_rows:
            if i_row >= max_rows:
                i_row += 1
                break

        i_row += 1


    print ('\nDone.  Read %d rows' % (i_row - 1))

    return selected_columns


#------------------------------------

def filter_datasets_interactive (bucket='all', client=None, save_search=False, restrict_key=True, restrict_value=False, dataset_oid_only=False, display_all_columns=False, max_rows=10):
    #TODO: This function has been replaced by 'search_files_interactive'.
    """This is an old way of searching for files. Not based on the current format. Only use

    Args:
        bucket (str or list, optional): buckets to search (defaults to searching all buckets you have access to in the datastore)

        client (optional): set client if not using the default

        restrict_key (bool, optional): if set to True, restricts the search to keys that are on the approved list (see file in bucket with dataset_key: accepted_key_values)

        restrict_key (bool, optional): if set to True, restricts the search to values that are on the approved list (see file in bucket with dataset_key: accepted_key_values)

        dataset_oid_only (bool, optional): if True, return a list of dataset_oids meeting the criteria;   if False, returns a dataframe of all the metadata for the files meeting search criteria

        display_all_columns (bool, optional): If 'False' (default), then show only a selected subset of the columns

        max_rows (int, optional): maximum rows to display during interactive search

    Returns:
        None

    """


    print("CAUTION: Use of filter_datasets_interactive is not recommended. This function has been replaced with 'search_files_interactive'. Please use 'search_files_interactive' instead")
    #configure client
    if client is None:
        client = config_client()

    # retrieve file with accepted keys and values if available (if user wants to restrict search to 'approved' keys:values)
    if restrict_key or restrict_value:
        try:
            kv_lookup = retrieve_dataset_by_datasetkey(bucket=bucket, dataset_key='accepted_key_values')
        except:
            print('Accepted_keys_values not defined for bucket(s) chosen. restrict_key and restrict_value will be set to False.')
            restrict_key = False
            restrict_value = False

    search_criteria = [bucket]
    # provide list of keys and have user select option
    print('Select a key from the following list:')
    keys = retrieve_keys(bucket = bucket)
    if restrict_key:
        approved_keys = kv_lookup.columns
        keys = list(set(keys) & set(approved_keys)) #display only keys that are both Approved and In use
    keys = sorted(keys, key = str.lower)

        #provides examples of types of values associated with the key to help users pick the key they want
    example_val_list=[]
    for key in keys:
        example_val = list(kv_lookup[key].unique())
        example_val_list.append(example_val)

        temp_dict={'value_examples': example_val_list, 'keys': keys, }
    display(pd.DataFrame.from_dict(temp_dict))
    key = input('Enter a key: ')

    # provide list of values and have user select option
    print("")
    print('Select value(s) for key=', key, 'from the following list: ')
    values_for_key = retrieve_values_for_key(key=key, bucket=bucket)
    if restrict_value:
        approved_values = list(kv_lookup[key].unique())
        values_for_key  = list(set(values_for_key ) & set(approved_values))
    print("")
    display(values_for_key)
    print("")
    value = input('Enter value(s) (comma separated for multiple values):  ')
    print(type(value))
    value = value.replace("'","")
    value = value.replace("[","")
    value = value.replace("]","")
    value = value.split(",")
    value = [x.strip(' ') for x in value]
    if type(values_for_key) == np.ndarray:
        value = [int(i) for i in value]
    #save key and value(s) searched
    search_criteria.append({'key': key, 'value': value, 'operator': "in"})

    # return dataframe of datasets meeting user specified criteria
    dataset_list = search_datasets_by_key_value(key=key, value=value, bucket=bucket, display_all_columns=display_all_columns)
    print('Number of datasets found meeting criteria =', len(dataset_list))
    if len(dataset_list) > max_rows:
        print('Displaying first %s results' %(max_rows))
    display(dataset_list.iloc[0:max_rows])

    if len(dataset_list) < 2:
        return dataset_list

    print("")
    repeat = input('Apply additional filter? (y/n)')

 #   if repeat == 'n':
  #      return dataset_list

    while repeat == 'y':
        key_values = list(dataset_list['metadata'])
        i = 0
        rows = len(key_values)
        while i < rows:
            new= key_values.pop(0)
            if i == 0:
                key_val = pd.DataFrame(new)
            else:
                key_val = pd.concat([key_val, new])
            i = i+1

        print('key_val size',key_val.shape[:])

        unique_keys = key_val['key'].unique()
        print("")
        print('Select a key from the following list:')
        if restrict_key:
            unique_keys = list(set(unique_keys) & set(approved_keys))
        labels = [('keys')]
        unique_keys = sorted(unique_keys, key = str.lower)

        example_val_list=[]
        for key in unique_keys:
            example_val = list(kv_lookup[key].unique())
            example_val_list.append(example_val)

        temp_dict={'value_examples': example_val_list, 'keys': unique_keys, }
        display(pd.DataFrame.from_dict(temp_dict))

        new_key = input('Enter a key: ')

        print("")
        print('Select value(s) for key=', new_key, 'from the following list: ')

        new_value = key_val[key_val['key'] == new_key]
        new_value = new_value['value'].unique()

        if restrict_value:
            approved_values = list(kv_lookup[new_key].unique())
            new_value  = list(set(new_value) & set(approved_values))
            print('if statement true')
        print("")
        display(new_value)
        print("")
        new_value = input('Enter value(s) (comma separated for multiple values):  ')
        new_value = new_value.replace("'","")
        new_value = new_value.replace(" ","")
        new_value = new_value.replace("  ","")
        new_value = new_value.split(",")
        new_value = [x.strip(' ') for x in new_value]
        print('values selected =', new_value, type(new_value))

        i = 0
        new_col = []

        #extract the values associated with the key
        while i < len(dataset_list):
            r = dataset_list.iloc[i]['metadata']
            keys = []
            for item in r:
                keys.append(item['value'])  #extracts keys from a
            keys = str(keys)
            keys = keys.replace("'","")
            keys = keys.replace("[","")
            keys = keys.replace("]","")

            if i == 0:
                new_col = [keys]
            else:
                new_col.append(keys)
            i += 1
        dataset_list['key_value'] = new_col

        dataset_list2 = dataset_list[dataset_list['key_value'].str.contains('|'.join(new_value)) ]
        #save key and value(s) searched
        search_criteria.append({'key': new_key, 'value': new_value, 'operator': "in"})

        print('Number of datasets found meeting criteria =', len(dataset_list2))
        if len(dataset_list2) > max_rows:
            print('Displaying first %s results' %(max_rows))
        display(dataset_list2.iloc[0:max_rows])

        print("")
        dataset_list = dataset_list2[:]
        print('--dataset_list length = ', len(dataset_list))

        if len(dataset_list) < 2:
            repeat = "n"

        repeat = input('Apply additional filter? (y/n)')

    if dataset_oid_only:
        return list(dataset_list['dataset_oid'])

    if save_search:
        print('search_criteria =', search_criteria)
        return search_criteria

    return dataset_list


#---------------------------------------------------
def summarize_datasets(dataset_keys, bucket, client=None, column=None, save_as=None, plot_ht=10, labels=None, last=False):
    """Generate summary statistics such as min/max/median/mean on files specified (all files must be in same bucket).

    Args:
        dataset_keys (list): dataset_keys corresponding to the files to summarize

        bucket (str): bucket the files reside in

        client (optional): set client if not using the default

        column (str, optional): column to summarize (will be prompted to specify if not pre-specified or if column does not exist in file)

        save_as (str, optional): filename to save image of box plot(s) to

        plot_ht (int, optional): height of box plots (default = 10)

        labels ('str', optional):

        last (bool optional): If True (default=False), then summarize values from last column instead of specifying column heading

    Returns
        (DataFrame): returns table summarizing the stats for the file(s) specified

    """
    import matplotlib.pyplot as plt

    if client is None:
        client = config_client()

    if type(dataset_keys) != list:
        dataset_keys = [dataset_keys]
    i=0

    check_col = column

    for key in dataset_keys:
        #retrieve the dataset as a pandas dataframe
        dataset = retrieve_dataset_by_datasetkey(dataset_key=key, bucket=bucket)

        # use the values in the last column
        if last:
            headers = list(dataset.columns)
            column = headers[-1]

        # provide a list of column names if column is not already specified
        if check_col is None:
            if column is not None:
                try:
                    d = dataset[column]
                except:
                    print(dataset.columns)

                    column = input("Pick a column from list to analyze: ")
            else:
                print(dataset.columns)

                column = input("Pick a column from list to analyze: ")


        # calculate stats for the column indicated
        d = dataset[column]
        d = pd.to_numeric(d)

        median = d.median()
        col_mode = d.mode()
        stats = d.describe()

        # combine stats into a summary table (pandas dataframe)
        stats = pd.DataFrame(stats)
        summary = [key, median, [col_mode]]
        summary = pd.DataFrame(summary, columns = [column], index = ['key', 'median', 'mode'])
        summary_temp = pd.concat([summary,stats])

        if i == 0:
            summary_table = summary_temp.rename(columns={column: '1'})
            data_to_plot = [d]
        else:
            summary_table[i+1] = summary_temp[column]
            data_to_plot.append(d)
        i += 1

    display(summary_table)

    # generate box and whisker plot

        # Create a figure instance
    fig = plt.figure(1, figsize=(3*len(dataset_keys), plot_ht))

        # Create an axes instance
    ax = fig.add_subplot(111)

        # Create the boxplot
    bp = ax.boxplot(data_to_plot)

    if labels:
        # Set x-labels for boxplot
        ax.set_xticklabels(labels)

    if save_as:
        fig.savefig(save_as, bbox_inches='tight')

    return summary_table

#----------------------------------------


def check_key_val(key_values, client=None, df=None, enforced=True):
    """Checks to ensure the keys and values specified are 'approved' and that (optionally) all required keys are filled out.

    Args:
        key_values (dict): keys and values specified by user for a file

        client (optional): set client if not using the default

        df (DataFrame): dataframe to be uploaded

        enforced (bool, optional): If True (default) checks that all required keys are filled out

    Returns:
        (bool): returns True if all keys and values are 'approved' AND enforcement criteria are met
    """

    if 'file_category' not in key_values:
        raise ValueError('file_category must be specified.')


    if client is None:
        client = config_client()

    #check if file_category is valid

    datasets = client.ds_datasets.get_datasets(dataset_key_regex='kv_lookup*', bucket_names=['default']).result()
    datasets = pd.DataFrame(datasets)
    i=0
    kv_lookup_dataset_keys = datasets['dataset_key']
    valid_file_category=[]
    while i < len(datasets):
        valid_file_category.append(kv_lookup_dataset_keys[i].replace('kv_lookup_',""))
        i+=1

    if key_values['file_category'] not in valid_file_category:
        raise ValueError('invalid file_category. Must be one of the following: %s' %valid_file_category)

    # generate dataset_key to retrieve the appropriate key:value lookup table
    file_cat = key_values['file_category']
    kv_lookup_dskey = ''.join(['kv_lookup_',file_cat])  #will need to enable to switch to auto-look up by category in default
    kv_lookup = retrieve_dataset_by_datasetkey(bucket='default', client=client, dataset_key=kv_lookup_dskey)
    #kv_lookup = pd.read_csv(kv_lookup_dskey+'.csv')

    # check that all keys are valid
    for key in key_values:
        if key not in kv_lookup:
            raise ValueError('key=%s invalid' %key,' Valid options include:', kv_lookup.iloc[:,0:-3].columns)

        # check that specified values are valid for given key
        if len(kv_lookup[key].unique()) > 1 :
            values = key_values.get(key)
            if type(values) != list:
                values = [values]
            for value in values:
                if any(kv_lookup[key] == value) != True:
                    raise ValueError('value=%s invalid' %value,'valid values for key=%s include:' %key, list(kv_lookup[key].unique()))

        # when applicable, check that the values input for id_col, smiles_col, and response_col are all headings that exist
        if df is not None :
            col_heading_req = ['id_col', 'smiles_col','response_col', 'parent_smiles_col']
            for col in col_heading_req:
                if col in list(key_values.keys()):
                    avail_headings = list(df.columns)
                    col_head_value = key_values.get(col)
                    if col_head_value not in avail_headings:
                        raise ValueError('value for key=%s invalid. Pick from these column headings:' %col, avail_headings)

    if enforced:
        """ This section checks to make sure all relevent keys have been filled in based on other selections made
             for example: if user includes 'curation_level':'ml_ready' as a key:value pair, then additional keys such as 'units' are also required
             1) this section requires the following 3 columns in the kv_lookup file: 'enforced_on_key', 'enforced_on_value', and 'required_keys'.
             2) if the 'enforced_on' key:value matches one input, then it checks to make sure all of the keys listed in the corresponding row in the
                'required_keys' column have been filled out """

        num_enforced_key = kv_lookup['enforced_on_key'].count()
        i=0

        while i < num_enforced_key:
            enforced_key = kv_lookup['enforced_on_key'][i]
            enforced_value = kv_lookup['enforced_on_value'][i]
            if enforced_key in key_values.keys():
                if enforced_value in key_values[enforced_key]:
                    required = (kv_lookup['required_keys'][i]).split(', ')
                    for key in required:
                        if key not in key_values:
                            raise ValueError('Required key missing: %s' %key)

            i += 1

#------------------------------------------------
def upload_file_to_DS(bucket, title, description, tags, key_values, filepath, filename, client=None, dataset_key=None, override_check=True, return_metadata=False, file_ref=False, data_type=None):
    """This function will upload a file to the Datastore along with the associated metadata

    Args:
        bucket (str): bucket the file will be put in

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): must be a list.

        key_values (dict): key:value pairs to enable future users to find the file. Must be a dictionary.

        filepath (str): current location of the file

        filename (str): current filename of the file

        client (optional): set client if not using the default

        dataset_key (str, optional): If updating a file already in the datastore enter the corresponding dataset_key.  If not, leave as 'none' and the dataset_key will be automatically generated.

        override_check (bool, optional): If 'True' then do NOT perform a check of the keys/values against approved list and enforcement criteria

        return_metadata (bool, optional): If 'True' (default=False), then return the metadata from the uploaded file

        file_ref (bool, optional): If 'True' (default=False), links file to the datastore instead of creating a copy to managed by the datastore.

        data_type (str,optional): Specify dataType (e.g. csv,bz, etc) if not specified attempt to use file extension

    Returns:
        (dict): optionally returns the metadata from the uploaded file (if return_metadata=True)
    """

    if client is None:
        client = config_client()

    filepath = os.path.join(filepath,filename)

    if type(key_values) != dict:
        raise ValueError('key_values must be a dictionary')

    if not override_check:
        ## JEA when pd is big, this will cause problems
        check_key_val(key_values=key_values, client=client, df=pd.read_csv(filepath))

    try:
        user = getpass.getuser()
    except:
        user = 'unknown'
    key_values.update({'user': user})

    key_values = json.dumps([key_values])

    #fileObj is what the datastore uploads
    #check file type
    split_file_ext = os.path.splitext(filepath)
    extension = split_file_ext[-1]
    #only open file if not creating a link
    if not file_ref :
        if extension == '.pkl':
            fileObj = open(filepath, 'rb')
        else:
            fileObj = io.FileIO(filepath)

    if dataset_key is None:
        dataset_key = filepath

    if not file_ref :
        req = client.ds_datasets.upload_dataset(
           bucket_name=bucket,
           title=title,
           description=description,
           tags=tags,
           metadata_obj=key_values,
           fileObj=fileObj,
           dataType=data_type,
           dataset_key=dataset_key,
           filename=filename,
        )
    else :
        req = client.ds_datasets.reference_dataset(
           bucket_name=bucket,
           title=title,
           description=description,
           tags=tags,
           metadata_obj=key_values,
           fileURL=filepath,
           dataType=data_type,
           dataset_key=dataset_key,
           filename=filename,
        )

    dataset = req.result()

    if return_metadata:
        return dataset

#------------------------------------------------
def upload_df_to_DS(df, bucket, filename, title, description, tags, key_values, client=None, dataset_key=None, override_check=True, return_metadata=False, index=False, data_type=None):
    """This function will upload a file to the Datastore along with the associated metadata

    Args:
        df (DataFrame): dataframe to be uploaded

        bucket (str): bucket the file will be put in

        filename (str): the filename to save the dataframe as in the datastore. Include the extension

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): must be a list.

        key_values (dict): key-value pairs to enable future users to find the file. Must be a dictionary.

        client (optional): set client if not using the default

        dataset_key (str): If updating a file already in the datastore enter the corresponding dataset_key.  If not, leave as 'none' and the dataset_key will be automatically generated.

        data_type (str,optional): Specify dataType (e.g. csv,bz, etc) if not specified attempt to use file extension

    Returns:
        (dict): if return_metadata=True, then function returns a dictionary of the metadata for the uploaded dataset.
    """

    if client is None:
        client = config_client()

    if type(key_values) != dict:
        raise ValueError('key_values must be a dictionary')

    if not override_check:
        check_key_val(key_values=key_values, client=client)


    df_shape = df.shape[:]
    num_row = df_shape[0]
    num_col = df_shape[1]
    try:
        user = getpass.getuser()
    except:
        user = 'unknown'
    key_values.update({'num_row':num_row, 'num_col':num_col, 'user': user})


    key_values = json.dumps([key_values])

    if '.csv' in filename:
        filename=filename
    elif '.' in filename:
        raise ValueError('filename extension must be .csv')
    else:
        filename = filename + '.csv'

    if dataset_key is None:
        dataset_key = bucket +'_'+ filename

    fileObj= df.to_csv(index=index)

    req = client.ds_datasets.upload_dataset(
       bucket_name=bucket,
       title=title,
       description=description,
       tags=tags,
       metadata_obj=key_values,
       fileObj=fileObj,
       dataset_key=dataset_key,
       dataType=data_type, 
       filename=filename,
    )
    dataset = req.result()

    if return_metadata:
        return dataset

#-----------------------------------------------
def update_kv(bucket, dataset_key, client=None, kv_add=None, kv_del=None, return_metadata=False):
    #TODO: function currently performs 2 separate uploads if adding and deleting, needs to be fixed to just 1 upload

    """update the key:values for specified file. No change to file.

    Args:
        bucket (str): Specify bucket where the file exists

        dataset_key (str): dataset_key for the file to update metadata for

        client (optional): set client if not using the default

        kv_add (dict, optional): key-value pairs to add to the metadata for the file specified

        kv_del (str or list, optional): keys to delete from the metadata for the file specified

    Returns:
        None

    """

    #configure client if needed
    if client is None:
        client = config_client()

    #check that bucket and dataset_key are valid
    if not dataset_key_exists(dataset_key=dataset_key, bucket=bucket, client=client):
        raise ValueError('dataset_key does not exist in bucket specified')

    # if kv_add is specified check to make sure format is right, then upload new keys:values
    if kv_add is not None:
        if type(kv_add) is not dict:
            raise ValueError('kv_add must be a dictionary')

        modified_dataset = {
                        'metadata': kv_add,
                        }

        results = client.ds_datasets.update_dataset(dataset_key=dataset_key, bucket_name=bucket, dataset=modified_dataset).result()

    # if kv_del is specified check to make sure format is right, then upload deletion of the keys specified
    if kv_del is not None:
        if type(kv_del) is not str and type(kv_del) is not list:
            raise ValueError('kv_del must be a string or list')

        if type(kv_del) is not list:
            kv_del = [kv_del]

        del_list = []
        for key in kv_del:
            del_list.append({'key': key, 'delete':True})

        modified_dataset = {
            'metadata': del_list
    }

        results = client.ds_datasets.update_dataset(dataset_key=dataset_key, bucket_name=bucket, dataset=modified_dataset).result()

    if return_metadata:
        return results
#----------------------------------------------

#-----------------------------------------------
def update_distribution_kv(bucket, dataset_key, client=None, kv_add=None, kv_del=None, return_metadata=False):
    #TODO: function currently performs 2 separate uploads if adding and deleting, needs to be fixed to just 1 upload
    #TODO: This should be merged with update_kv()
    """update the key:values for specified file. No change to file.

    Args:
        bucket (str): Specify bucket where the file exists

        dataset_key (str): dataset_key for the file to update metadata for

        client (optional): set client if not using the default

        kv_add (dict, optional): key-value pairs to add to the metadata for the file specified

        kv_del (str or list, optional): keys to delete from the metadata for the file specified

    Returns:
        None

    """

    #configure client if needed
    if client is None:
        client = config_client()

    #check that bucket and dataset_key are valid
    if not dataset_key_exists(dataset_key=dataset_key, bucket=bucket, client=client):
        raise ValueError('dataset_key does not exist in bucket specified')

    # if kv_add is specified check to make sure format is right, then upload new keys:values
    if kv_add is not None:
        if type(kv_add) is not dict:
            raise ValueError('kv_add must be a dictionary')

        modified_dataset = {
                        'distribution': kv_add,
                        }

        results = client.ds_datasets.update_dataset(dataset_key=dataset_key, bucket_name=bucket, dataset=modified_dataset).result()

    # if kv_del is specified check to make sure format is right, then upload deletion of the keys specified
    if kv_del is not None:
        if type(kv_del) is not str and type(kv_del) is not list:
            raise ValueError('kv_del must be a string or list')

        if type(kv_del) is not list:
            kv_del = [kv_del]

        del_list = []
        for key in kv_del:
            del_list.append({'key': key, 'delete':True})

        modified_dataset = {
            'metadata': del_list
    }

        results = client.ds_datasets.update_dataset(dataset_key=dataset_key, bucket_name=bucket, dataset=modified_dataset).result()

    if return_metadata:
        return results
#----------------------------------------------

def repeat_defined_search(defined_search, client=None, to_return='df', display_all_columns=False):
    """Retrieves a DataFrame of files (and associated metadata) meeting the search criteria.
    This is designed to work well with the output from the filter_datasets_interactive function with defined_search=True

    Args:
        defined_search (list): a list with position 0 = string/list of buckets, and remaining positions dictionaries of search criteria
            example: defined_search = ['gsk_ml',
                {'key': 'species', 'value': ['rat'], 'operator': 'in'},
                {'key': 'assay_category','value': ['solubility', 'volume_of_distribution'], 'operator': 'in'}]

        client (optional): set client if not using the default
        to_return (str, optional): (default=df)
                'df' (df_results)  = return a pandas dataframe summarizing metadata of files meeting criteria
                oid' (dataset_oid) = return a list of dataset_oids meeting criteria
                ds_key' (dataset_key) = return a list of dataset_key + bucket tuples

        display_all_column (bool, optional): default False. If True, displays all associated metadata instead of just a selected subset

    Returns:
        One of the following will be returned (based on selection for 'to_return')
        (DataFrame): dataframe of metadata for the files matching the criteria specified in the search
        (list): list of dataset_oids meeting the criteria specified in the search
        (list): list of bucket and dataset_key meeting the criteria specified in the search
    """

    if to_return not in ['df', 'oid','ds_key']:
        raise ValueError('to_return entry invalid')

    bucket = defined_search[0]
    key_val_criteria = json.dumps(defined_search[1:])

    if client is None:
        client = config_client()

    #search for files meeting criteria
    files = client.ds_datasets.get_datasets(metadata=key_val_criteria, bucket_name=bucket).result()

    files = pd.DataFrame(files)

    if len(files) == 0:
        print('No files found matching criteria specified')
    else:
        if not display_all_columns:
            col = ['bucket_name', 'title', 'dataset_oid', 'dataset_key', 'description',
                   'metadata', 'tags', 'user_perm', 'active', 'versions']
            files = files[col]

    if to_return == 'df': #(df_results)
        return files

    if to_return == 'oid': #(dataset_oid)
        return list(files['dataset_oid'])

    if to_return == 'ds_key': #(dataset_key)
        return list(zip(files['bucket_name'], files['dataset_key']))


#----------------------------------------------------------------
def get_keyval(dataset_oid=None, dataset_key=None, bucket=None, client=None):
    """Requires either dataset_oid *or* dataset_key+bucket.
       Function extracts the key:value pairs and converts from the 'datastore format' (list of dictionaries) into 'model tracker format' (a single dictionary).
    """


    if client is None:
        client = config_client()

    # check that dataset_oid *or* dataset_key+bucket was entered
    if dataset_oid:
        if dataset_key:
            raise ValueError('Both dataset_oid and dataset_key are specified.')
        ds_metadata = retrieve_dataset_by_dataset_oid(dataset_oid=dataset_oid, return_metadata=True, client=client)
    if dataset_key:
        ds_metadata = retrieve_dataset_by_datasetkey(bucket=bucket, dataset_key=dataset_key, return_metadata=True, client=client)
    if not dataset_oid:
        if not dataset_key:
            raise ValueError('dataset_oid or dataset_key + bucket required')

    # convert
    ds_metadata = ds_metadata['metadata']
    kv_pairs = len(ds_metadata)
    i = 0
    new_dict = {}

    while i < kv_pairs:
        kv = ds_metadata[i]
        key = kv.get('key')
        value = kv.get('value')
        new_dict.update({key: value})
        i+=1

    return new_dict

#------------
def upload_pickle_to_DS(data, bucket, filename, title, description, tags, key_values,client=None, dataset_key=None, override_check=True, return_metadata=False):
    """This function will upload a file to the Datastore along with the associated metadata.

    Args:
        data (DataFrame, str, list, tuple, pickle): data to be pickled and uploaded

        bucket (str): bucket the file will be put in

        filename (str): the filename to save the dataframe as in the datastore. Include the extension

        title (str): title of the file in (human friendly format)

        description (str): long text box to describe file (background/use notes)

        tags (list): must be a list.

        key_values (dict): key:value pairs to enable future users to find the file. Must be a dictionary.

        client (optional): set client if not using the default

        dataset_key (str, optional): If updating a file already in the datastore enter the corresponding dataset_key.
                          If not, leave as 'none' and the dataset_key will be automatically generated.

        override_check (bool, optional): If True, overrides checking the metadata for the file when uploaded.

        return_metadata (bool, optional): If True, returns metadata for the file after it is uploaded.

    Returns:
        None

    """

    if client is None:
        client = config_client()

    if type(key_values) != dict:
        raise ValueError('key_values must be a dictionary')

    if not override_check:
        check_key_val(key_values=key_values, df=df, client=client)

    try:
        user = getpass.getuser()
    except:
        user = 'unknown'
    key_values.update({'user': user})


    key_values = json.dumps([key_values])

    if dataset_key is None:
        dataset_key = bucket +'_'+ filename

    if type(data) != bytes:
        fileObj= pickle.dumps(data)
    else:
        fileObj = data

    req = client.ds_datasets.upload_dataset(
       bucket_name=bucket,
       title=title,
       description=description,
       tags=tags,
       metadata_obj=key_values,
       fileObj=fileObj,
       dataset_key=dataset_key,
       filename=filename,
    )
    dataset = req.result()

    if return_metadata:
        return dataset

    #------------------------------------
def list_key_values(bucket, input_key, category='experimental', client=None):
    #TODO:
    """List the values for input key.  Requires that the input key be in the 'approved' list

    Args:
        bucket (str or list, optional): buckets to search (defaults to searching all buckets you have access to in the datastore)

        input_key: user specified key to query

        category: 'experimental' or 'pdb_bind'

        client (optional): set client if not using the default

    Returns:
        None

    """

    values_for_key=[]

    if client is None :
        client = config_client()

    if key_exists(input_key, bucket, client) :

        #retrieve lookup table
        dataset_key = 'kv_lookup_'+ category
        kv_lookup = retrieve_dataset_by_datasetkey(bucket='default', dataset_key=dataset_key)
        if input_key in kv_lookup :

            values_for_key = retrieve_values_for_key(key=input_key, bucket=bucket)
        else :
            print("Error Key not on approved list",input_key,kv_lookup)
    return values_for_key

    #------------------------------------
def search_files_interactive (bucket='all', client=None, to_return='df', display_all_columns=False, max_rows=10):
    #TODO: This will replace filter_datasets_interactive eventually. This function uses the new key:value lookup tables
    """This tool helps you find the files you need via an interactive/guided interface.

    Args:
        bucket (str or list, optional): buckets to search (defaults to searching all buckets you have access to in the datastore)

        client (optional): set client if not using the default

        to_return (str): 'df' (df_results)  = return a pandas dataframe summarizing metadata of files meeting criteria
                         'search' (search_criteria) = return a list containing search criteria where position 0 = string/list of buckets, and remaining positions are dictionaries of search criteria.
                                                      Designed to work with 'repeat_defined_search' function.
                         'oid' (dataset_oid) = return a list of dataset_oids meeting criteria
                         'ds_key' (dataset_key) = return a list of dataset_key + bucket tuples

        display_all_columns (bool, optional): If 'False' (default), then show only a selected subset of the columns

        max_rows (int, optional): maximum rows to display during interactive search

    Returns:
        None

    """

    if to_return not in ['df','search','oid','ds_key']:
        raise ValueError('to_return entry invalid')

    #configure client
    if client is None:
        client = config_client()

    # determine file category
    file_categories = retrieve_values_for_key(key='kv_lookup', bucket="default")
    category=""
    if len(file_categories) == 1:
        category = file_categories[0]
    while category not in file_categories:
        print('Select file category. Options: ', file_categories)
        category = input('Enter a selection: ')

    #retrieve lookup table
    dataset_key = 'kv_lookup_'+ category
    kv_lookup = retrieve_dataset_by_datasetkey(bucket='default', dataset_key=dataset_key)

    search_criteria = [bucket, {'key':'file_category', 'value':[category], 'operator':'in'}]  # used for saving the search for easy retrieval, updated as selections are made
    used_keys = ['file_category']

    # provide list of keys and have user select option
    print('Select a key from the following list:')
    keys = retrieve_keys(bucket = bucket)
    approved_keys = kv_lookup.columns
        #display only keys that 1) exist, 2) are approved (in kv_lookup table), and 3) not already used. Then sort (ascending).
    keys = list(set(keys) & set(approved_keys))
    for key in used_keys:
        keys.remove(key)
    keys = sorted(keys, key = str.lower)

    #provides examples of types of values associated with the key to help users pick the key they want
    example_val_list=[]
    for key in keys:
        example_val = list(kv_lookup[key].unique())
        example_val_list.append(example_val)
        temp_dict={'value_examples': example_val_list, 'keys': keys, }
    print(pd.DataFrame.from_dict(temp_dict))
    input_key = ""
    while input_key not in keys:
        input_key = input('Enter a key: ')
    used_keys.append(input_key)

    # provide list of values and have user select option
    print("")
    print('Select value(s) for key=', input_key, 'from the following list: ')
    values_for_key = retrieve_values_for_key(key=input_key, bucket=bucket)
    print("")
    print(values_for_key)
    print("")

    values_valid=False
    operator='in'
    while values_valid == False:
        invalid_value = False
        value = input('Enter value(s) (comma separated for multiple values):  ')

        value = value.replace("'","")
        value = value.replace("[","")
        value = value.replace("]","")
        value = value.split(",")
        value = [x.strip(' ') for x in value]
        print('currently value=', value)  #delete?
        if type(values_for_key) == np.ndarray:
            if '>=' in value[0]:
                operator='>='
                value = value[0].replace(">=","")
                value = values_for_key[np.where(values_for_key >= int(value))]
            elif '<=' in value[0]:
                operator='<='
                value = value[0].replace("<=","")
                value = values_for_key[np.where(values_for_key <= int(value))]
            elif '>' in value[0]:
                operator='>'
                value = value[0].replace(">","")
                value = values_for_key[np.where(values_for_key > int(value))]
            elif '<' in value[0]:
                operator='<'
                value = value[0].replace("<","")
                value = values_for_key[np.where(values_for_key < int(value))]
            else:
                value = [int(i) for i in value]
        for value in value:
            if value not in values_for_key:
                invalid_value = True
                print('value %s is not valid ' %value)
        if invalid_value == False:
            values_valid = True

    #save key and value(s) searched
    search_criteria.append({'key': input_key, 'value': [value], 'operator': "in"})

    # return dataframe of datasets meeting user specified criteria
    dataset_list = search_datasets_by_key_value(key=input_key, value=value, operator=operator, bucket=bucket, display_all_columns=display_all_columns)
    print('Number of datasets found meeting criteria =', len(dataset_list))
    if len(dataset_list) > max_rows:
        print('Displaying first %s results' %(max_rows))
    print(dataset_list.iloc[0:max_rows])

    if len(dataset_list) < 2:
        return dataset_list

    print("")

    repeat = ""
    while repeat not in ['y','n']:
        repeat = input('Apply additional filter? (y/n)')

    #-----refine search section ----
    while repeat == 'y':
        key_values = list(dataset_list['metadata'])
        i = 0
        rows = len(key_values)
        while i < rows:
            new= key_values.pop(0)
            if i == 0:
                key_val = pd.DataFrame(new)
            else:
                key_val = key_val.append(new)
            i = i+1

        unique_keys = key_val['key'].unique()
        print("")
        print('Select a key from the following list:')
        unique_keys = list(set(unique_keys) & set(approved_keys))
        for key in used_keys:
            unique_keys.remove(key)
        unique_keys = sorted(unique_keys, key = str.lower)
        example_val_list=[]
        for key in unique_keys:
            example_val = list(kv_lookup[key].unique())
            example_val_list.append(example_val)
        temp_dict={'value_examples': example_val_list, 'keys': unique_keys, }
        print(pd.DataFrame.from_dict(temp_dict))

        new_key=""
        while new_key not in approved_keys:
            new_key = input('Enter a key: ')
        used_keys.append(new_key)

        print("")
        print('Select value(s) for key=', new_key, 'from the following list: ')

        values_for_key = key_val[key_val['key'] == new_key]
        values_for_key = values_for_key['value'].unique()

        approved_values = list(kv_lookup[new_key].unique())
        values_for_key  = list(set(values_for_key) & set(approved_values))

        print("")
        print(values_for_key)
        print("")
        ##
        values_valid=False
        while values_valid == False:
            invalid_value = False
            new_value = input("Enter value(s) (comma separated for multiple values) or Enter 'change key' to change the key':  ")
            if new_value == "change key":
                    used_keys.pop()
                    print("")
                    print('Select a key from the following list:')
                    unique_keys = list(set(unique_keys) & set(approved_keys))
                    unique_keys = sorted(unique_keys, key = str.lower)
                    example_val_list=[]
                    for key in unique_keys:
                        example_val = list(kv_lookup[key].unique())
                        example_val_list.append(example_val)
                    temp_dict={'value_examples': example_val_list, 'keys': unique_keys, }
                    print(pd.DataFrame.from_dict(temp_dict))

                    new_key=""
                    while new_key not in approved_keys:
                        new_key = input('Enter a key: ')
                    used_keys.append(new_key)

                    print("")
                    print('Select value(s) for key=', new_key, 'from the following list: ')

                    values_for_key = key_val[key_val['key'] == new_key]
                    values_for_key = values_for_key['value'].unique()

                    approved_values = list(kv_lookup[new_key].unique())
                    values_for_key  = list(set(values_for_key) & set(approved_values))

                    print("")
                    print(values_for_key)
                    print("")

            new_value = new_value.replace("'","")
            new_value = new_value.replace("[","")
            new_value = new_value.replace("]","")
            new_value = new_value.split(",")
            new_value = [x.strip(' ') for x in new_value]
            if type(values_for_key) == np.ndarray:
                new_value = [int(i) for i in new_value]
            for value in new_value:
                if value not in values_for_key:
                    invalid_value = True
                    print('value %s is not valid ' %value)
            if invalid_value == False:
                values_valid = True
        print('values selected =', new_value, type(new_value))

        i = 0
        new_col = []

        #extract the values associated with the key
        while i < len(dataset_list):
            r = dataset_list.iloc[i]['metadata']
            keys = []
            for item in r:
                keys.append(item['value'])  #extracts keys from a
            keys = str(keys)
            keys = keys.replace("'","")
            keys = keys.replace("[","")
            keys = keys.replace("]","")

            if i == 0:
                new_col = [keys]
            else:
                new_col.append(keys)
            i += 1
        dataset_list['key_value'] = new_col

        dataset_list2 = dataset_list[dataset_list['key_value'].str.contains('|'.join(new_value)) ]

        #save key and value(s) searched
        search_criteria.append({'key': new_key, 'value': new_value, 'operator': "in"})

        print('Number of datasets found meeting criteria =', len(dataset_list2))
        if len(dataset_list2) > max_rows:
            print('Displaying first %s results' %(max_rows))
        print(dataset_list2.iloc[0:max_rows])

        print("")
        dataset_list = dataset_list2[:]
        print('--dataset_list length = ', len(dataset_list))

        if len(dataset_list) < 2:
            repeat = "n"

        repeat = ""
        while repeat not in ['y','n']:
            repeat = input('Apply additional filter? (y/n)')

    if to_return == 'df': #(df_results)
        return dataset_list

    if to_return == 'search': #(search_criteria)
        print('search_criteria =', search_criteria)
        return search_criteria

    if to_return == 'oid': #(dataset_oid)
        return list(dataset_list['dataset_oid'])

    if to_return == 'ds_key': #(dataset_key)
        return list(zip(dataset_list['bucket_name'], dataset_list['dataset_key']))


#--------------------------------------------------------------------
def bulk_export_kv_for_files(files, save_as, client=None):
    #TODO: function is slow. look into speeding up.
    """exports a csv file with 3 columns: bucket, dataset_key, key/value pairs to make reviewing  metadata easier

    Args:
        files (list of tuples): format [(bucket1, dataset_key1), (bucket2, dataset_key2)]

        save_as (str): filename to use for new file

    Returns:
        None

    """

    #configure client
    if client is None:
        client = config_client()

    if type(files) is not list:
        raise ValueError(" 'files' must be a list")

    file_list = []
    summary = []
    for item in files:
        if type(item) is not tuple:
            raise ValueError("each item in 'files' must be a tuple formatted (bucket, dataset_key)")
        bucket=item[0]
        dataset_key=item[1]
        metadata = get_keyval(bucket=bucket, dataset_key=dataset_key, client=client)
        file_list = [bucket, dataset_key, metadata]
        summary.append(file_list)

    summary = pd.DataFrame(summary)
    summary.to_csv(save_as)

#----------------------------------------------------------------------
#NOTE: Bulk update keys/values function uses this function

def string_to_dict(dict_string):
    # Convert to proper json format
    dict_string = dict_string.replace("'", '"').replace('u"', '"')
    return json.loads(dict_string)

#---------------------------------------------------------------------
#NOTE: Bulk update keys/values function uses this function
def string_to_list(list_string):
    # Convert to proper json format
    list_string = list_string.replace("'", '').replace("[","").replace("]","").replace(",","").replace('u"', '"')
    list_string=list_string.split()
    return list_string

#---------------------------------------------------------------------
### upload info bor files to change
def bulk_update_kv(file, client=None, i=0):
    """this function allows you to upload a properly formatted csv file with 4 columns (order and spelling of headings must match): bucket, dataset_key, kv_add, kv_del
    the metadata for the files listed will then be updated in the Datastore
    """

    #configure client
    if client is None:
        client = config_client()

    # import file
    edit_files = pd.read_csv(file)

    #check headings
    req_headings = ['bucket', 'dataset_key', 'kv_add', 'kv_del']
    cols_used = list(edit_files.columns)
    if cols_used != req_headings:
        raise ValueError('headings need to match approved format: bucket, dataset_key, kv_add, kv_del')

     #loop through files and update metadata for each

    i = i
    while i < len(edit_files):
        row = list(edit_files.iloc[i])
        bucket = row[0]
        dataset_key=row[1]
        kv_add=row[2]
        print('i =', i, 'dataset_key = ', dataset_key)
        if type(kv_add) == str:
            kv_add=string_to_dict(kv_add)
            len_add = len(kv_add)
        else:
            len_add = 0
        kv_del=row[3]
        if type(kv_del) == str:
            kv_del=string_to_list(kv_del)
            len_del = len(kv_del)
        else:
            len_del = 0
        i += 1

        if len_add > 0 and len_del > 0:
            update_kv(bucket, dataset_key, kv_add=kv_add, kv_del=kv_del, client=client)

        elif len_add > 0 and len_del == 0:
            update_kv(bucket, dataset_key, kv_add=kv_add, client=client)

        elif len_del > 0 and len_add == 0:
             update_kv(bucket, dataset_key, kv_del=kv_del, client=client)

#---------------------------------------------------------------------



def get_key_val(metadata, key=None):
    """Simple utility to search through list of key value pairs and return values for query key

    Args:
        metadata list of key,value pairs (list): a list with position 0 = string/list of buckets, and remaining positions dictionaries of search criteria
                              example:
                                 [{'key': 'species', 'value': ['rat'] },
                                 {'key': 'assay_category','value': ['solubility', 'volume_of_distribution']}]

       key (str):  key to search for

    Returns:
        returns When key is provide, returns value for matching key if found, None otherwise
        returns dictionary for the list of key,value pairs when a query key is not provided.
    """
    if key == None :
        return dict([(kv['key'], kv['value']) for kv in metadata])
    else :
        ret_val=None
        for elem in metadata :
            if elem['key'] == key :
                ret_val=elem['value']
                break
        return ret_val

#---------------------------------------------------------------------
def copy_datasets_to_bucket(dataset_keys, from_bucket, to_bucket, ds_client=None):
    """Copy each named dataset from one bucket to another.

    Args:
        dataset_keys (str or list of str): List of dataset_keys for datasets to move.

        from_bucket (str): Bucket where datasets are now.

        to_bucket (str): Bucket to move datasets to.

    Returns:
        None

    """
    if ds_client is None:
        ds_client = config_client()

    if type(dataset_keys) == str:
        dataset_keys = [dataset_keys]

    # Copy each dataset
    for dataset_key in dataset_keys:
        try:
            ds_meta = ds_client.ds_datasets.copy_dataset(bucket_name=from_bucket, dataset_key=dataset_key, to_bucket_name=to_bucket).result()
        except Exception as e:
            print('Error copying dataset %s\nfrom bucket %s to %s' % (dataset_key, from_bucket, to_bucket))
            print(e)
            continue
        print('Copied dataset %s to %s' % (dataset_key, to_bucket))


