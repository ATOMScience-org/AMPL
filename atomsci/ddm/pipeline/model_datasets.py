"""Classes for dealing with datasets for data-driven modeling."""

import logging
import os
import shutil
from deepchem.data import NumpyDataset
import numpy as np
import pandas as pd
import deepchem as dc
import uuid
from atomsci.ddm.pipeline import featurization as feat
from atomsci.ddm.pipeline import splitting as split
from atomsci.ddm.utils import datastore_functions as dsf
from pathlib import Path
import getpass
import sys

feather_supported = True
try:
    import pyarrow.feather as feather
except (ImportError, AttributeError, ModuleNotFoundError):
    feather_supported = False

logging.basicConfig(format='%(asctime)-15s %(message)s')


# ****************************************************************************************
def create_model_dataset(params, featurization, ds_client=None):
    """Factory function for creating DatastoreDataset or FileDataset objects.

    Args:
        params (Namespace object): contains all parameter information.

        featurization (Featurization object): The featurization object created by ModelDataset or input as an argument
        in the factory function.

        ds_client (Datastore client)

    Returns:
        either (DatastoreDataset) or (FileDataset): instantiated ModelDataset subclass specified by params
    """
    if params.datastore:
        return DatastoreDataset(params, featurization, ds_client)
    else:
        return FileDataset(params, featurization)

# ****************************************************************************************
def create_minimal_dataset(params, featurization, contains_responses=False):
    """Create a MinimalDataset object for non-persistent data (e.g., a list of compounds or SMILES
    strings or a data frame). This object will be suitable for running predictions on a pretrained
    model, but not for training.

    Args:
        params (Namespace object): contains all parameter information.

        featurization (Featurization object): The featurization object created by ModelDataset or input as an argument
        in the factory function.

        contains_responses (Boolean): Boolean specifying whether the dataset has a column with response values

    Returns:
        (MinimalDataset): a new MinimalDataset object

    """
    return MinimalDataset(params, featurization, contains_responses)

# ****************************************************************************************
def check_task_columns(params, dset_df):
    """Check that the data frame dset_df contains all the columns listed in params.response_cols.

    Args:
        params (Namespace): Parsed parameter structure; must include response_cols parameter at minimum.

        dset_df (pd.DataFrame): Dataset as a DataFrame that contains columns for the prediction tasks

    Raises:
        Exception:
            If response columns not set in params.
            If input dataset is missing response columns

    """
    if params.response_cols is None:
        raise Exception("Unable to determine prediction tasks for dataset")
    missing_tasks = list(set(params.response_cols) - set(dset_df.columns.values))
    if len(missing_tasks) > 0:
        raise Exception(f"Requested prediction task columns {missing_tasks} are missing from training dataset")


# ****************************************************************************************
def set_group_permissions(system, path, data_owner='public', data_owner_group='public'):
    """Set file group and permissions to standard values for a dataset containing proprietary
    or public data, as indicated by 'data_owner'.

    Args:
        system (string): Determine the group ownership (at the moment 'LC', 'AD')

        path (string): File path

        data_owner (string): Who the data belongs to, either 'public' or the name of a company (e.g. 'gsk') associated
        with a restricted access group.
            'username': group is set to the current user's username
            'data_owner_group': group is set to data_owner_group
            Otherwise, group is set by hard-coded dictionary.

    Returns:
        None
    """

    # Currently, if we're not on an LC machine, we're on an AD-controlled system. This could change.
    if system != 'LC':
        system = 'AD'

    owner_group_map = dict(public={'LC': 'atom', 'AD': 'atom'})

    if data_owner == 'username':
        group = getpass.getuser()
    elif data_owner == 'data_owner_group':
        group = data_owner_group
    else:
        group = owner_group_map[data_owner][system]

    try:
        path_metadata = Path(path)
        # TODO: MJT I have this if statement to deal with the permission errors on /ds/projdata. May not be necessary.
        if path_metadata.group() != group:
            shutil.chown(path, group=group)
            os.chmod(path, 0o770)
    except FileNotFoundError:
        # On LC, it seems that os.walk() can return files that are pending removal
        pass

# ****************************************************************************************
def key_value_list_to_dict(kvp_list):
    """Convert a key-value pair list from the datastore metadata into a proper dictionary

    Args:
        kvp_list (list): List of key-value pairs

    Returns:
        (dictionary): the kvp-list reformatted as a dictionary
    """
    return dict([(d['key'], d['value']) for d in kvp_list])

# ***************************************************************************************
# TODO: This function refers to hardcoded directories on TTB. It is only called by model_pipeline.regenerate_results.
# Consider moving this function outside the pipeline code.
def create_split_dataset_from_metadata(model_metadata, ds_client, save_file=False):
    """Function that pulls the split metadata from the datastore and then joins that info with the dataset itself.
    Args:
        model_metadata (Namespace): Namespace object of model metadata

        ds_client: datastore client

        save_file (Boolean): Boolean specifying whether we want to save split dataset to disk

    Returns:
        (DataFrame): DataFrame with subset information and response column
    """
    split_uuid = model_metadata.split_uuid
    dset_key = model_metadata.dataset_key
    bucket = model_metadata.bucket
    print(dset_key)
    try:
        split_metadata = dsf.search_datasets_by_key_value('split_dataset_uuid', split_uuid, ds_client, operator='in',
                                                          bucket=bucket)
        split_oid = split_metadata['dataset_oid'].values[0]
        split_df = dsf.retrieve_dataset_by_dataset_oid(split_oid, client=ds_client)
    except Exception as e:
        print("Error when loading split file:\n%s" % str(e))
        return None
    try:
        dataset_df = dsf.retrieve_dataset_by_datasetkey(dset_key, bucket, client=ds_client)
    except Exception as e:
        print("Error when loading dataset:\n%s" % str(e))
        return None
    try:
        #TODO Need to investigate why these are not the same length
        #print(len(set(dataset_df['compound_id'].values.tolist()) - set(split_df['cmpd_id'].values.tolist())))
        joined_dataset = dataset_df.merge(split_df, how='inner', left_on='compound_id', right_on='cmpd_id').drop('cmpd_id', axis=1)
    except Exception as e:
        print("Error when joining dataset with split dataset:\n%s" % str(e))
        return None
    if save_file:
        shortened_key = dset_key.rstrip('.csv')
        res_path = '/ds/projdata/gsk_data/split_joined_datasets'
        filename = '%s_%s_%s_split_dataset.csv' % (shortened_key, split_uuid, model_metadata.splitter)
        joined_dataset.to_csv(os.path.join(res_path, filename), index=False)
    return joined_dataset

# ***************************************************************************************
# TODO: This function isn't used anywhere; consider removing it.
def save_joined_dataset(joined_dataset, split_metadata):
    """DEPRECATED: Refers to absolute file paths that no longer exist.

    Args:
        joined_dataset (DataFrame): DataFrame containing split information with the response column

        split_metadata (dictionary): Dictionary containing metadata with split info

    Returns:
        None
    """
    print(split_metadata['metadata'])
    dset_key = split_metadata['dataset_key']
    split_uuid = split_metadata['metadata']['split_uuid']
    res_path = '/ds/projdata/gsk_data/split_joined_datasets'
    filename = '%s_%s_split_dataset.csv' % (dset_key, split_uuid)
    joined_dataset.to_csv(os.path.join(res_path, filename), index=False)
    """
    title = 'Joined dataset for dataset %s and split UUID %s' % (dset_key, split_uuid)
    description = "Joined dataset dset_key %s with split labels for split UUID %s" % (dset_key, split_uuid)
    # TODO: Need a way to differentiate between original file and this
    tags = split_metadata['tags']
    key_values = split_metadata['metadata']
    # dfs.upload_df_to_DS(joined_dataset, bucket, filename, title, description, tags, key_values, client=ds_client)
    """

# ****************************************************************************************
def get_classes(y):
    """Returns the class indices for a set of labels"""
    
    # remove nan's automatically flattens the array
    # even if there are no nan's in it
    class_indeces = set(y[~np.isnan(y)])

    return class_indeces

# ****************************************************************************************

class ModelDataset(object):
    """Base class representing a dataset for data-driven modeling. Subclasses are specialized for dealing with
    dataset objects persisted in the datastore or in the filesystem.

    Attributes:

        set in __init__:
            params (Namespace object): contains all parameter information

            log (logger object): logger for all warning messages.

            dataset_name (str): set from the parameter object, the name of the dataset

            output_dit (str): The root directory for saving output files

            split_strategy (str): the flag for determining the split strategy (e.g. 'train_test_valid','k-fold')

            featurization (Featurization object): The featurization object created by ModelDataset or input as an
            argument in the factory function.

            splitting (Splitting object): A splitting object created by the ModelDataset intiailization method

            combined_train_valid_data (dc.Dataset): A dataset object (initialized as None), of the merged train
            and valid splits

        set in get_featurized_data:
            dataset: A new featurized DeepChem Dataset.

            n_features: The count of features (int)

            vals: The response col after featurization (np.array)

            attr: A pd.dataframe containing the compound ids and smiles

        set in get_dataset_tasks:
            tasks (list): list of prediction task columns

        set in split_dataset or load_presplit_dataset:
            train_valid_dsets:  A list of tuples of (training,validation) DeepChem Datasets

            test_dset: (dc.data.Dataset): The test dataset to be held out

            train_valid_attr: A list of tuples of (training,validation) attribute DataFrames

            test_attr: The attribute DataFrame for the test set, containing compound IDs and SMILES strings.
    """

    def __init__(self, params, featurization):
        """Initializes ModelDataset object.

        Arguments:
            params (Namespace object): contains all parameter information.

            featurization: Featurization object; will be created if necessary based on params

        """
        self.params = params
        self.log = logging.getLogger('ATOM')
        self.dataset_name = self.params.dataset_name
        self.dataset_oid = None
        self.output_dir = self.params.output_dir
        # By default, a ModelDataset contains both features and response data. Subclasses can
        # override this.
        self.contains_responses = True

        # Create object to delegate featurization to
        if featurization is None:
            self.featurization = feat.create_featurization(self.params)
        else:
            # Reuse existing Featurization object
            self.featurization = featurization

        if self.params.previously_split and self.params.split_uuid is None:
            raise Exception(
                    "previously_split is set to True but no split_uuid provided for dataset {}".
                    format(self.dataset_name))
        if not self.params.previously_split and self.params.split_uuid is not None:
            self.log.info("previously_split is set False; ignoring split_uuid passed as parameter")
            self.params.split_uuid = None
        if self.params.split_uuid is None:
            self.split_uuid = str(uuid.uuid4())
        else:
            self.split_uuid = self.params.split_uuid
        # Defer creating the splitting object until we know if we're using a previous split or a new one
        self.splitting = None

        # Cache for combined training and validation data, used by k-fold CV code
        self.combined_train_valid_data = None
        # Cache for subset-specific response values matched to IDs, used by k-fold CV code
        self.subset_response_dict = {}
        # Cache for subset-specific response values matched to IDs, used by k-fold CV code
        self.subset_weight_dict = {}

    # ****************************************************************************************
    def load_full_dataset(self):
        """Loads the dataset from the datastore or the file system.

        Raises:
            NotImplementedError: The method is implemented by subclasses
        """
        raise NotImplementedError

    # ****************************************************************************************
    def load_featurized_data(self):
        """Loads prefeaturized data from the datastore or filesystem. Returns a data frame,
        which is then passed to featurization.extract_prefeaturized_data() for processing.

        Raises:
            NotImplementedError: The method is implemented by subclasses

        """
        raise NotImplementedError

    # ****************************************************************************************
    def get_featurized_data(self, params=None):
        """Does whatever is necessary to prepare a featurized dataset.
        Loads an existing prefeaturized dataset if one exists and if parameter previously_featurized is
        set True; otherwise loads a raw dataset and featurizes it. Creates an associated DeepChem Dataset
        object.

        Side effects:
            Sets the following attributes in the ModelDataset object:
                dataset: A new featurized DeepChem Dataset.
                n_features: The count of features (int)
                vals: The response col after featurization (np.array)
                attr: A pd.dataframe containing the compound ids and smiles
        """
        
        if params is None:
            params = self.params
        if params.previously_featurized:
            try:
                self.log.debug("Attempting to load featurized dataset")
                featurized_dset_df = self.load_featurized_data()
                if (params.max_dataset_rows > 0) and (len(featurized_dset_df) > params.max_dataset_rows):
                    featurized_dset_df = featurized_dset_df.sample(n=params.max_dataset_rows)
                featurized_dset_df[params.id_col] = featurized_dset_df[params.id_col].astype(str)
                self.log.debug("Got dataset, attempting to extract data")
                features, ids, self.vals, self.attr = self.featurization.extract_prefeaturized_data(
                                                           featurized_dset_df, params)
                self.n_features = self.featurization.get_feature_count()
                self.log.debug("Creating deepchem dataset")
                
                # don't do make_weights which convert all NaN rows into 0 for hybrid model
                if params.model_type != "hybrid":
                    self.vals, w = feat.make_weights(self.vals, is_class=params.prediction_type=='classification')
                else:
                    w = np.ones_like(self.vals)

                if params.prediction_type=='classification':
                    w = w.astype(np.float32)

                self.dataset = NumpyDataset(features, self.vals, ids=ids, w=w)
                self.log.info("Using prefeaturized data; number of features = " + str(self.n_features))
                return
            except AssertionError as a:
                raise a
            except Exception as e:
                self.log.debug("Exception when trying to load featurized data:\n%s" % str(e))
                self.log.info("Featurized dataset not previously saved for dataset %s, creating new" % self.dataset_name)
                pass
        else:
            self.log.info("Creating new featurized dataset for dataset %s" % self.dataset_name)
        dset_df = self.load_full_dataset()
        sample_only = False
        if (params.max_dataset_rows > 0) and (len(dset_df) > params.max_dataset_rows):
            dset_df = dset_df.sample(n=params.max_dataset_rows).reset_index(drop=True)
            sample_only = True
        check_task_columns(params, dset_df)
        features, ids, self.vals, self.attr, w, featurized_dset_df = self.featurization.featurize_data(dset_df, params, self.contains_responses)
        if not sample_only:
            self.save_featurized_data(featurized_dset_df)

        self.n_features = self.featurization.get_feature_count()
        self.log.debug("Number of features: " + str(self.n_features))
           
        # Create the DeepChem dataset       
        self.dataset = NumpyDataset(features, self.vals, ids=ids, w=w)
        # Checking for minimum number of rows
        if len(self.dataset) < params.min_compound_number:
            self.log.warning("Dataset of length %i is shorter than the required length %i" % (len(self.dataset), params.min_compound_number))

    # ****************************************************************************************
    def get_dataset_tasks(self, dset_df):
        """Sets self.tasks to the list of prediction task (response) columns defined by the current model parameters.

        Args:
            dset_df (pd.DataFrame): Dataset as a DataFrame that contains columns for the prediction tasks

        Returns:
            sucess (boolean): True is self.tasks is set. False if not user supplied.

        Side effects:
            Sets the self.tasks attribute to be the list of prediction task columns
        """

        # Superclass method just looks in params; is called by subclasses (therefore, should NOT raise an exception if this fails)
        self.tasks = None
        if self.params.response_cols is not None:
            if type(self.params.response_cols) == list:
                self.tasks = self.params.response_cols
            else:
                self.tasks = [self.params.response_cols]
        return self.tasks is not None

    # ****************************************************************************************
    def split_dataset(self):
        """Splits the dataset into paired training/validation and test subsets, according to the split strategy
                selected by the model params. For traditional train/valid/test splits, there is only one training/validation
                pair. For k-fold cross-validation splits, there are k different train/valid pairs; the validation sets are
                disjoint but the training sets overlap.

        Side effects:
           Sets the following attributes in the ModelDataset object:
               train_valid_dsets:  A list of tuples of (training,validation) DeepChem Datasets

               test_dset: (dc.data.Dataset): The test dataset to be held out

               train_valid_attr: A list of tuples of (training,validation) attribute DataFrames

               test_attr: The attribute DataFrame for the test set, containing compound IDs and SMILES strings.
        """

        # Create object to delegate splitting to.
        if self.splitting is None:
            self.splitting = split.create_splitting(self.params)
        self.train_valid_dsets, self.test_dset, self.train_valid_attr, self.test_attr = \
            self.splitting.split_dataset(self.dataset, self.attr, self.params.smiles_col)
        if self.train_valid_dsets is None:
            raise Exception("Dataset %s did not split properly" % self.dataset_name)
        if self.params.prediction_type == 'classification':
            self._validate_classification_dataset()

    # ****************************************************************************************

    def _validate_classification_dataset(self):
        """Verifies that this is a valid data for classification
        Checks that all classes are represented in all subsets. This causes performance metrics to crash.
        Checks that multi-class labels are between 0 and class_number
        """
        if not self._check_classes():
            raise ClassificationDataException("Dataset {} does not have all classes represented in a split".format(self.dataset_name))
        if not self._check_deepchem_classes():
            raise ClassificationDataException("Dataset {} does not have all classes labeled using positive integers 0 <= i < {}".format(self.dataset_name, self.params.class_number))

    def _check_classes(self):
        """Checks to see if all classes are represented in all splits.

        Returns:
            (Boolean): boolean specifying if all classes are specified in all splits
        """
        ref_class_set = get_classes(self.train_valid_dsets[0][0].y)
        for train, valid in self.train_valid_dsets:
            if not ref_class_set == get_classes(train.y):
                return False
            if not ref_class_set == get_classes(valid.y):
                return False

        if not ref_class_set == get_classes(self.test_dset.y):
            return False
        return True

    # ****************************************************************************************

    def _check_deepchem_classes(self):
        """Checks if classes adhear to DeepChem class index convention. Classes must be >=0 and < class_number

        Returns:
            (Boolean): boolean spechifying if classes adhear to DeepChem convention
        """
        classes = get_classes(self.dataset.y)
        return all([0 <= c < self.params.class_number for c in list(classes)])

    # ****************************************************************************************

    def get_split_metadata(self):
        """Creates a dictionary of the parameters related to dataset splitting, to be saved in the model tracker
        along with the other metadata needed to reproduce a model run.

        Returns:
            dict: A dictionary containing the data needed to reproduce the current dataset training/validation/test
            splits, including the lists of compound IDs for each split subset
        """
        split_data = dict(split_uuid = self.split_uuid)
        for param in split.split_params:
            split_data[param] = self.params.__dict__.get(param, None)
        return split_data

    # ****************************************************************************************
    def create_dataset_split_table(self):
        """Generates a data frame containing the information needed to reconstruct the current
        train/valid/test or k-fold split.

        Returns:
            split_df (DataFrame): Table with one row per compound in the dataset, with columns:
                cmpd_id:    Compound ID

                subset:     The subset the compound was assigned to in the split. Either 'train', 'valid',
                            'test', or 'train_valid'. 'train_valid' is used for a k-fold split to indicate
                            that the compound was rotated between training and validation sets.

                fold:       For a k-fold split, an integer indicating the fold in which the compound was in
                            the validation set. Is zero for compounds in the test set and for all compounds
                            when a train/valid/test split was used.
        """
        ids = []
        subsets = []
        folds = []
        # ksm: Consider moving this method to the Splitting class
        if self.params.split_strategy == 'k_fold_cv':
            for fold, (train_attr, valid_attr) in enumerate(self.train_valid_attr):
                nvalid = valid_attr.shape[0]
                ids = ids + valid_attr.index.values.tolist()
                subsets += ['train_valid'] * nvalid
                folds += [fold] * nvalid
        else:
            train_attr, valid_attr = self.train_valid_attr[0]
            ntrain = train_attr.shape[0]
            ids = ids + train_attr.index.values.tolist()
            subsets += ['train'] * ntrain

            nvalid = valid_attr.shape[0]
            ids = ids + valid_attr.index.values.tolist()
            subsets += ['valid'] * nvalid
            folds += [0] * (ntrain+nvalid)

        ntest = self.test_attr.shape[0]
        ids = ids + self.test_attr.index.values.tolist()
        subsets += ['test'] * ntest
        folds += [0] * ntest

        split_df = pd.DataFrame(dict(cmpd_id=ids, subset=subsets, fold=folds))
        split_df = split_df.drop_duplicates(subset='cmpd_id')
        return split_df

    # ****************************************************************************************
    def load_presplit_dataset(self, directory=None):
        """Loads a table of compound IDs assigned to split subsets, and uses them to split
        the currently loaded featurized dataset.

        Args:
            directory (str): Optional directory where the split table is stored; used only by FileDataset.

            Defaults to the directory containing the current dataset.

        Returns:
            success (boolean): True if the split table was loaded successfully and used to split the dataset.

        Side effects:
                Sets the following attributes of the ModelDataset object
                    train_valid_dsets:  A list of tuples of (training,validation) DeepChem Datasets

                    test_dset: (dc.data.Dataset): The test dataset to be held out

                    train_valid_attr: A list of tuples of (training,validation) attribute DataFrames

                    test_attr: The attribute DataFrame for the test set, containing compound IDs and SMILES strings.
        Raises:
            Exception: Catches exceptions from split.select_dset_by_attr_ids
                            or from other errors while splitting dataset using metadata
        """

        # Load the split table from the datastore or filesystem
        self.splitting = split.create_splitting(self.params)

        try:
            split_df, split_kv = self.load_dataset_split_table(directory)
        except Exception as e:
            self.log.error("Error when loading dataset split table:\n%s" % str(e))
            sys.exit(1)

        self.train_valid_attr = []
        self.train_valid_dsets = []

        # Override the splitting-related model parameters, if they differ from the ones stored with the split table
        if split_kv is not None:
            for param in split.split_params:
                if param in split_kv:
                    if self.params.__dict__[param] != split_kv[param]:
                        self.log.warning("Warning: %s = %s in split table, %s in model params; using split table value." %
                                         (param, str(split_kv[param]), str(self.params.__dict__[param])))
                        self.params.__dict__[param] = split_kv[param]

        # Create object to delegate splitting to.
        if self.params.split_strategy == 'k_fold_cv':
            train_valid_df = split_df[split_df.subset == 'train_valid']
            for f in range(self.splitting.num_folds):
                train_df = train_valid_df[train_valid_df.fold != f]
                valid_df = train_valid_df[train_valid_df.fold == f]
                train_dset = split.select_dset_by_id_list(self.dataset, train_df.cmpd_id.astype(str).values)
                valid_dset = split.select_dset_by_id_list(self.dataset, valid_df.cmpd_id.astype(str).values)
                train_attr = split.select_attrs_by_dset_ids(train_dset, self.attr)
                valid_attr = split.select_attrs_by_dset_ids(valid_dset, self.attr)
                self.train_valid_dsets.append((train_dset, valid_dset))
                self.train_valid_attr.append((train_attr, valid_attr))
        else:
            train_df = split_df[split_df.subset == 'train']
            valid_df = split_df[split_df.subset == 'valid']
            train_dset = split.select_dset_by_id_list(self.dataset, train_df.cmpd_id.astype(str).values)
            valid_dset = split.select_dset_by_id_list(self.dataset, valid_df.cmpd_id.astype(str).values)
            train_attr = split.select_attrs_by_dset_ids(train_dset, self.attr)
            valid_attr = split.select_attrs_by_dset_ids(valid_dset, self.attr)
            self.train_valid_dsets.append((train_dset, valid_dset))
            self.train_valid_attr.append((train_attr, valid_attr))

        test_df = split_df[split_df.subset == 'test']
        self.test_dset = split.select_dset_by_id_list(self.dataset, test_df.cmpd_id.astype(str).values)
        self.test_attr = split.select_attrs_by_dset_ids(self.test_dset, self.attr)

        self.log.warning('Previous dataset split restored')
        return True


    # ****************************************************************************************

    def combined_training_data(self):
        """Returns a DeepChem Dataset object containing data for the combined training & validation compounds.

        Returns:
            combined_dataset (dc.data.Dataset): Dataset containing the combined training
            and validation data.

        Side effects:
            Overwrites the combined_train_valid_data attribute of the ModelDataset with the combined data
        """
        # All of the splits have the same combined train/valid data, regardless of whether we're using
        # k-fold or train/valid/test splitting.
        if self.combined_train_valid_data is None:
            (train, valid) = self.train_valid_dsets[0]
            combined_X = np.concatenate((train.X, valid.X), axis=0)
            combined_y = np.concatenate((train.y, valid.y), axis=0)
            combined_w = np.concatenate((train.w, valid.w), axis=0)
            combined_ids = np.concatenate((train.ids, valid.ids))
            self.combined_train_valid_data = NumpyDataset(combined_X, combined_y, w=combined_w, ids=combined_ids)
        return self.combined_train_valid_data

    # ****************************************************************************************

    def has_all_feature_columns(self, dset_df):
        """Compare the columns in dataframe dset_df against the feature columns required by
        the current featurization and descriptor_type param. Returns True if dset_df contains
        all the required columns.

        Args:
            dset_df (DataFrame): Feature matrix

        Returns:
            (Boolean): boolean specifying whether there are any missing columns in dset_df
        """
        missing_cols = set(self.featurization.get_feature_columns()) - set(dset_df.columns.values)
        return (len(missing_cols) == 0)

    # *************************************************************************************

    def get_subset_responses_and_weights(self, subset, transformers):
        """Returns a dictionary mapping compound IDs in the given dataset subset to arrays of response values
        and weights.  Used by the perf_data module under k-fold CV.

        Args:
            subset (string): Label of subset, 'train', 'test', or 'valid'

            transformers: Transformers object for full dataset

        Returns:
            tuple(response_dict, weight_dict)
                (response_dict): dictionary mapping compound ids to arrays of per-task untransformed response values
                (weight_dict): dictionary mapping compound ids to arrays of per-task weights
        """
        if subset not in self.subset_response_dict:
            if subset in ('train', 'valid', 'train_valid'):
                dataset = self.combined_training_data()
            elif subset == 'test':
                dataset = self.test_dset
            else:
                raise ValueError('Unknown dataset subset type "%s"' % subset)

            y = dc.trans.undo_transforms(dataset.y, transformers)
            w = dataset.w
            response_vals = dict([(id, y[i,:]) for i, id in enumerate(dataset.ids)])
            weights = dict([(id, w[i,:]) for i, id in enumerate(dataset.ids)])
            self.subset_response_dict[subset] = response_vals
            self.subset_weight_dict[subset] = weights
        return self.subset_response_dict[subset], self.subset_weight_dict[subset]

    # *************************************************************************************

    def _get_split_key(self):
        """Creates the proper CSV name for a split file

        Returns:
            (str): String containing the dataset name, split type, and split_UUID. Used as key in datastore or filename
            on disk.
        """
        return '{0}_{1}_{2}.csv'.format(self.dataset_name, self.splitting.get_split_prefix(), self.split_uuid)

# ****************************************************************************************

class MinimalDataset(ModelDataset):
    """A lightweight dataset class that does not support persistence or splitting, and therefore can be
    used for predictions with an existing model, but not for training a model. Is not expected to
    contain response columns, i.e. the ground truth is assumed to be unknown.

        Attributes:

        set in __init__:
            params (Namespace object): contains all parameter information
            log (logger object): logger for all warning messages.
            featurization (Featurization object): The featurization object created by ModelDataset or input as an optional argument in the factory function.

        set in get_featurized_data:
            dataset: A new featurized DeepChem Dataset.
            n_features: The count of features (int)
            attr: A pd.dataframe containing the compound ids and smiles

        set in get_dataset_tasks:
            tasks (list): list of prediction task columns

    """

    def __init__(self, params, featurization, contains_responses=False):
        """Initializes MinimalDataset object.

        Args:
            params (Namespace object): contains all parameter information.

            featurization: Featurization object. Unlike in other ModelDataset subclasses, the featurization object must
            be provided at creation time.

        """

        self.params = params
        self.log = logging.getLogger('ATOM')
        self.featurization = featurization
        self.dataset = None
        self.n_features = None
        self.tasks = None
        self.attr = None
        self.contains_responses = contains_responses

    # ****************************************************************************************
    def get_dataset_tasks(self, dset_df):
        """Sets self.tasks to the list of prediction task columns defined for this dataset. These should be defined in
        the params.response_cols list that was provided when this object was created.

        Args:
            dset_df (pd.DataFrame): Ignored in this version.

        Returns:
            Success (bool): Returns true if task names are retrieved.

        Side effects:
            Sets the task attribute of the MinimalDataset object to a list of task names.
        """
        return super().get_dataset_tasks(dset_df)

    # ****************************************************************************************
    def get_featurized_data(self, dset_df, is_featurized=False):
        """Featurizes the compound data provided in data frame dset_df, and creates an
        associated DeepChem Dataset object.

        Args:
            dset_df (DataFrame): DataFrame either with compound id and smiles string or Feature matrix

            is_featurized (Boolean): boolean specifying whether the dset_df is already featurized

        Returns:
            None

        Side effects:
            Sets the following attributes in the ModelDataset object:

                dataset: A new featurized DeepChem Dataset.

                n_features: The count of features (int)

                attr: A pd.dataframe containing the compound ids and smiles
        """

        params = self.params
        if is_featurized:
            # Input data frame already contains feature columns
            self.log.warning("Formatting already featurized data...")
            feature_cols = [dset_df[col].values.reshape(-1,1) for col in self.featurization.get_feature_columns()]
            features = np.concatenate(feature_cols, axis=1)
            ids = dset_df[params.id_col].astype(str).values
            #TODO: check size is right
            nrows = len(ids)
            ncols = len(params.response_cols)
            if self.contains_responses:
                self.vals = dset_df[params.response_cols].values
            else:
                self.vals = np.zeros((nrows,ncols))
            self.attr = pd.DataFrame({params.smiles_col: dset_df[params.smiles_col].values},
                                 index=dset_df[params.id_col])
            self.log.warning("Done")
        else:
            self.log.warning("Featurizing data...")
            features, ids, self.vals, self.attr, weights, featurized_dset_df  = self.featurization.featurize_data(dset_df, 
                                                                                    params, self.contains_responses)
            self.log.warning("Done")
        self.n_features = self.featurization.get_feature_count()
        self.dataset = NumpyDataset(features, self.vals, ids=ids)

    # ****************************************************************************************
    def save_featurized_data(self, featurized_dset_df):
        """Does nothing, since a MinimalDataset object does not persist its data.

        Args:
                featurized_dset_df (pd.DataFrame): Ignored.
        """

# ****************************************************************************************

class DatastoreDataset(ModelDataset):
    """Subclass representing a dataset for data-driven modeling that lives in the datastore.

        Attributes:

        set in __init__:
            params (Namespace object): contains all parameter information

            log (logger object): logger for all warning messages.

            dataset_name (str): set from the parameter object, the name of the dataset

            output_dit (str): The root directory for saving output files

            split_strategy (str): the flag for determining the split strategy (e.g. 'train_test_valid','k-fold')

            featurization (Featurization object): The featurization object created by ModelDataset or input as an optional argument in the factory function.

            splitting (Splitting object): A splitting object created by the ModelDataset intiailization method

            combined_train_valid_data (dc.Dataset): A dataset object (initialized as None), of the merged train and valid splits
            ds_client (datastore client):

        set in get_featurized_data:
            dataset: A new featurized DeepChem Dataset.

            n_features: The count of features (int)

            vals: The response col after featurization (np.array)

            attr: A pd.dataframe containing the compound ids and smiles

        set in get_dataset_tasks:
            tasks (list): list of prediction task columns

        set in split_dataset or load_presplit_dataset:
            train_valid_dsets:  A list of tuples of (training,validation) DeepChem Datasets

            test_dset: (dc.data.Dataset): The test dataset to be held out

            train_valid_attr: A list of tuples of (training,validation) attribute DataFrames

            test_attr: The attribute DataFrame for the test set, containing compound IDs and SMILES strings.

        set in load_full_dataset()
            dataset_key (str): The datastore key pointing to the dataset

    """

    #TODO: Added featurization=None as default, is this ok?
    def __init__(self, params, featurization=None, ds_client=None):
        """Initializes DatastoreDataset object.

        Args:
            params (Namespace object): contains all parameter information.

            ds_client: datastore client.

            featurization: Featurization object; will be created if necessary based on params

        """

        super().__init__(params, featurization)
        self.dataset_oid = None
        if params.dataset_name:
            self.dataset_name = params.dataset_name
        else:
            self.dataset_name = os.path.basename(self.params.dataset_key).replace('.csv', '')
        if ds_client is None:
            self.ds_client = dsf.config_client()
        else:
            self.ds_client = ds_client

    # ****************************************************************************************
    def load_full_dataset(self):
        """Loads the dataset from the datastore

        Returns:
            dset_df: Dataset as a DataFrame

        Raises:
            Exception if dset_df is None or empty due to error in loading in dataset.
        """
        self.dataset_key = self.params.dataset_key
        dset_df = dsf.retrieve_dataset_by_datasetkey(self.dataset_key, self.params.bucket, self.ds_client)
        dset_df[self.params.id_col] = dset_df[self.params.id_col].astype(str)
        dataset_metadata = dsf.retrieve_dataset_by_datasetkey(self.dataset_key, self.params.bucket, self.ds_client,
                                                          return_metadata=True)
        self.dataset_oid = dataset_metadata['dataset_oid']

        if dset_df is None:
            raise Exception("Failed to load dataset %s" % self.dataset_key)
        if dset_df.empty:
            raise Exception("Dataset %s is empty" % self.dataset_key)
        return dset_df

    # ****************************************************************************************
    def get_dataset_tasks(self, dset_df):
        """Sets self.tasks to the list of prediction task columns defined for this dataset. If the dataset is in the datastore,
        these should be available in the metadata. Otherwise we guess by looking at the column names in dset_df
        and excluding features, compound IDs, SMILES string columns, etc.

        Args:
            dset_df (pd.DataFrame): Dataset containing the prediction tasks

        Returns:
            Success (bool): Returns true if task names are retrieved.

        Side effects:
            Sets the task attribute of the DatastoreDataset object to a list of task names.
        """
        # TODO (ksm): Can we get rid of this function and just call the superclass version (which gets the tasks from response_cols)?
        # response_cols is a required parameter now, so there's no need for guessing.

        if super().get_dataset_tasks(dset_df):
            return True
        else:
            # If not specified by user, get tasks from dataset info
            try:
                dataset_info = self.ds_client.ds_datasets.get_bucket_dataset(bucket_name=self.params.bucket,
                                                                             dataset_key=self.dataset_key).result()
                keyval_dict = key_value_list_to_dict(dataset_info['metadata'])
                task_name = keyval_dict['task_name']
                self.tasks = [task_name]
                return True
            except Exception:
                # Try to guess tasks from data frame columns, assuming that tasks are anything that doesn't look
                # like a compound ID, SMILES string, other compound identifier, or feature
                non_task_cols = {'compound_id', 'rdkit_smiles', self.params.id_col, self.params.smiles_col, 'inchi_key',
                                 'inchi_string', 'smiles', 'smiles_out', 'lost_frags'} | set(
                    self.featurization.get_feature_columns())
                self.tasks = sorted(set(dset_df.columns.values) - non_task_cols)
        if self.tasks is None or not self.tasks:
            self.log.error("Unable to determine prediction task(s) for dataset %s" % self.dataset_name)
            return False
        return True

    # ****************************************************************************************

    def save_featurized_data(self, featurized_dset_df):
        """Save a featurized dataset to the datastore

        Args:
            featurized_dset_df: DataFrame containing the featurized dataset.

        Returns:
            None
        """
        featurized_dset_name = self.featurization.get_featurized_dset_name(self.dataset_name)
        featurized_dset_key = os.path.join(os.path.dirname(self.params.dataset_key), featurized_dset_name)

        # Add metadata key/value pairs from source object, if it came from datastore
        dataset_info = self.ds_client.ds_datasets.get_bucket_dataset(bucket_name=self.params.bucket,
                                                                     dataset_key=self.dataset_key).result()
        keyval_dict = key_value_list_to_dict(dataset_info['metadata'])
        # Give the current user credit for creating the prefeaturized dataset
        keyval_dict['user'] = os.environ['USER']
        # Add some key/value pairs to indicate the parent dataset and featurization used
        keyval_dict['parent_dataset_key'] = self.dataset_key
        keyval_dict['parent_dataset_bucket'] = self.params.bucket
        keyval_dict['parent_dataset_oid'] = self.dataset_oid
        keyval_dict['featurization'] = str(self.featurization)
        # Get the tags from the original dataset, and add some more to indicate this is a prefeaturized
        # dataset, plus some clues to its provenance
        tag_list = dataset_info['tags']
        tag_list += ['prefeaturized', self.dataset_name.lower(), self.params.featurizer.lower()]

        dataset_metadata = dsf.upload_df_to_DS(featurized_dset_df,
                           bucket=self.params.bucket,
                           filename=featurized_dset_key,
                           title=featurized_dset_key.replace('_', ' '),
                           description='Data from dataset %s featurized with %s' % (
                           self.dataset_name, str(self.featurization)),
                           tags=tag_list,
                           key_values=keyval_dict,
                           client=self.ds_client,
                           dataset_key=featurized_dset_key,
                           return_metadata=True)
        # Set the OID for the featurized dataset
        self.dataset_oid = dataset_metadata['dataset_oid']

    # ****************************************************************************************
    def load_featurized_data(self):
        """Loads prefeaturized data from the datastore. Returns a data frame,
        which is then passed to featurization.extract_prefeaturized_data() for processing.

        Returns:
            featurized_dset_df (pd.DataFrame): dataframe of the prefeaturized data, needs futher processing
        """
        # If a dataset OID for a specific featurized dataset was provided, use it
        if self.params.dataset_oid is not None:
            # Check the tags for this OID to make sure it's really the correct prefeaturized dataset.
            metadata = dsf.retrieve_dataset_by_dataset_oid(self.params.dataset_oid, self.ds_client,
                             return_metadata=True)
            tags = metadata['tags']
            if ('prefeaturized' in tags and self.dataset_name.lower() in tags 
                                        and self.params.featurizer.lower() in tags):
                featurized_dset_df = dsf.retrieve_dataset_by_dataset_oid(self.params.dataset_oid, self.ds_client)
                if self.has_all_feature_columns(featurized_dset_df):
                    self.dataset_oid = self.params.dataset_oid
                    self.dataset_key = metadata['dataset_key']
                    return featurized_dset_df
        # Check to see if the dataset specified by params.dataset_key is already featurized
        metadata = dsf.retrieve_dataset_by_datasetkey(self.params.dataset_key, self.params.bucket,
                                                      self.ds_client, return_metadata=True)
        if 'prefeaturized' in metadata['tags']:
            self.log.debug("Loading prefeaturized dataset from key %s, bucket %s" % (
                           self.params.dataset_key, self.params.bucket))
            featurized_dset_df = dsf.retrieve_dataset_by_datasetkey(self.params.dataset_key, 
                                                                    self.params.bucket, 
                                                                    self.ds_client)
            # Check that dataframe has columns needed for descriptor type
            if self.has_all_feature_columns(featurized_dset_df):
                self.dataset_key = self.params.dataset_key
                self.dataset_oid = metadata['dataset_oid']
                return featurized_dset_df
        # Otherwise, look it up by key and bucket
        featurized_dset_name = self.featurization.get_featurized_dset_name(self.dataset_name)
        featurized_dset_key = os.path.join(os.path.dirname(self.params.dataset_key), featurized_dset_name)
        self.log.debug("Looking up prefeaturized dataset at key %s, bucket %s" % (featurized_dset_key, self.params.bucket))
        featurized_dset_df = dsf.retrieve_dataset_by_datasetkey(featurized_dset_key, self.params.bucket, self.ds_client)
        self.dataset_key = featurized_dset_key

        # Look up the OID for the featurized dataset. We'll need this later when we save the model metadata.
        featurized_dset_metadata = dsf.retrieve_dataset_by_datasetkey(featurized_dset_key, self.params.bucket, 
                                                                      self.ds_client, return_metadata=True)
        self.dataset_oid = featurized_dset_metadata['dataset_oid']
        featurized_dset_df[self.params.id_col] = featurized_dset_df[self.params.id_col].astype(str)
        return featurized_dset_df

    # ****************************************************************************************
    def save_split_dataset(self, directory=None):
        """Saves a table of compound IDs assigned to each split subset of a dataset.

        Args:
            directory:    Ignored; included only for compatibility with the FileDataset version of this method.
        """
        split_df = self.create_dataset_split_table()
        split_table_key = self._get_split_key()

        keyval_dict = dict(user = os.environ['USER'], full_dataset_key = self.dataset_key,
                           full_dataset_bucket = self.params.bucket, full_dataset_oid = self.dataset_oid,
                           split_dataset_uuid = self.split_uuid,
                           file_category= 'ml_data_split')
        for param in split.split_params:
            try:
                keyval_dict[param] = self.params.__dict__[param]
            except KeyError:
                pass

        tag_list = ['split_table']

        dsf.upload_df_to_DS(split_df,
                           bucket=self.params.bucket,
                           filename=split_table_key,
                           title="Split table %s" % split_table_key.replace('_', ' '),
                           description='Dataset %s %s split compound assignment table' % (
                                        self.dataset_name, self.splitting.get_split_prefix()),
                           tags=tag_list,
                           key_values=keyval_dict,
                           client=self.ds_client,
                           dataset_key=split_table_key)
        print("save split info",split_table_key,self.params.bucket,self.split_uuid)
        self.log.info('Dataset split_uuid = %s' % self.split_uuid)
        self.log.info('Dataset split table saved to datastore bucket %s with dataset_key %s' % (self.params.bucket,
                      split_table_key))

    # ****************************************************************************************
    def load_dataset_split_table(self, directory=None):
        """Loads from the datastore a table of compound IDs assigned to each split subset of a dataset.
        Called by load_presplit_dataset().

        Args:
            directory:    Ignored; included only for compatibility with the FileDataset version of this method.

        Returns:
            tuple(split_df, split_kv):
                split_df (DataFrame): Table assigning compound IDs to split subsets and folds.
                split_kv (dict): Dictionary of key-value pairs from the split table metadata; includes all the
                                 parameters that were used to define the split.
        """
        split_metadata = dsf.search_datasets_by_key_value('split_dataset_uuid', self.params.split_uuid, self.ds_client,
                                                          operator='in',
                                                          bucket=self.params.bucket)
        split_oid = split_metadata['dataset_oid'].values[0]
        split_df = dsf.retrieve_dataset_by_dataset_oid(split_oid, client=self.ds_client)
        split_meta = dsf.retrieve_dataset_by_dataset_oid(split_oid, client=self.ds_client, return_metadata=True)
        split_kv = dsf.get_key_val(split_meta['metadata'])
        return split_df, split_kv



# ****************************************************************************************


class FileDataset(ModelDataset):
    """Subclass representing a dataset for data-driven modeling that lives in the filesystem.

    Attributes:

        set in __init__:
            params (Namespace object): contains all parameter information

            log (logger object): logger for all warning messages.

            dataset_name (str): set from the parameter object, the name of the dataset

            output_dit (str): The root directory for saving output files

            split_strategy (str): the flag for determining the split strategy (e.g. 'train_test_valid','k-fold')

            featurization (Featurization object): The featurization object created by ModelDataset or input as an
            optional argument in the factory function.

            splitting (Splitting object): A splitting object created by the ModelDataset intiailization method

            combined_train_valid_data (dc.Dataset): A dataset object (initialized as None), of the merged train and
            valid splits

            ds_client (datastore client):

        set in get_featurized_data:
            dataset: A new featurized DeepChem Dataset.

            n_features: The count of features (int)

            vals: The response col after featurization (np.array)

            attr: A pd.dataframe containing the compound ids and smiles

        set in get_dataset_tasks:
            tasks (list): list of prediction task columns

        set in split_dataset or load_presplit_dataset:
            train_valid_dsets:  A list of tuples of (training,validation) DeepChem Datasets

            test_dset: (dc.data.Dataset): The test dataset to be held out

            train_valid_attr: A list of tuples of (training,validation) attribute DataFrames

            test_attr: The attribute DataFrame for the test set, containing compound IDs and SMILES strings.
    """

    def __init__(self, params, featurization):
        """Initializes FileDataset object.

        Args:
            params (Namespace object): contains all parameter information.

            featurization: Featurization object; will be created if necessary based on params
        """
        super().__init__(params, featurization)
        if params.dataset_name:
            self.dataset_name = params.dataset_name
        else:
            self.dataset_name = os.path.basename(self.params.dataset_key).replace('.csv', '')

    # ****************************************************************************************
    def load_full_dataset(self):
        """Loads the dataset from the file system.

        Returns:
            dset_df: Dataset as a DataFrame loaded in from a CSV or feather file

        Raises:
            exception: if dataset is empty or failed to load
        """
        dataset_path = self.params.dataset_key
        if not os.path.exists(dataset_path):
            raise Exception("Dataset file %s does not exist" % dataset_path)
        if dataset_path.endswith('.feather'):
            if not feather_supported:
                raise Exception("feather package not installed in current environment")
            dset_df = feather.read_dataframe(dataset_path)
        elif dataset_path.endswith('.csv'):
            dset_df = pd.read_csv(dataset_path, index_col=False)
        else:
            raise Exception('Dataset %s is not a recognized format (csv or feather)' % dataset_path)

        if dset_df is None:
            raise Exception("Failed to load dataset %s" % dataset_path)
        if dset_df.empty:
            raise Exception("Dataset %s is empty" % dataset_path)
        dset_df[self.params.id_col] = dset_df[self.params.id_col].astype(str)
        return dset_df

    # ****************************************************************************************

    def get_dataset_tasks(self, dset_df):
        """Returns the list of prediction task columns defined for this dataset. If the dataset is in the datastore,
        these should be available in the metadata. Otherwise you can guess by looking at the column names in dset_df
        and excluding features, compound IDs, SMILES string columns, etc.

        Args:
            dset_df (pd.DataFrame): Dataset as a DataFrame that contains columns for the prediction tasks

        Returns:
            sucess (boolean): True is self.tasks is set. False if not user supplied.

        Side effects:
            Sets the self.tasks attribute of FileDataset to be the list of prediction task columns
        """
        if super().get_dataset_tasks(dset_df):
            return True
        else:
            # Guess: Tasks are anything that doesn't look like a compound ID, SMILES string, other compound identifier,
            # or feature
            non_task_cols = {'compound_id', 'rdkit_smiles', self.params.id_col, self.params.smiles_col, 'inchi_key',
                             'inchi_string', 'smiles', 'smiles_out', 'lost_frags'} | set(
                self.featurization.get_feature_columns())
            self.tasks = sorted(set(dset_df.columns.values) - non_task_cols)
        if self.tasks is None or not self.tasks:
            self.log.error("Unable to determine prediction task(s) for dataset %s" % self.dataset_name)
            return False
        return True

    # ****************************************************************************************

    def save_featurized_data(self, featurized_dset_df):
        """Save a featurized dataset to the filesystem.

        Args:
            featurized_dset_df (pd.DataFrame): Dataset as a DataFrame that contains the featurized data
        """

        try:
            featurized_dset_name = self.featurization.get_featurized_dset_name(self.dataset_name)
        except NotImplementedError:
            # Featurizer is non-persistent, so do nothing
            return

        dataset_dir = os.path.dirname(self.params.dataset_key)
        data_dir = os.path.join(dataset_dir, self.featurization.get_featurized_data_subdir())
        featurized_dset_path = os.path.join(data_dir, featurized_dset_name)

        if os.path.isfile(featurized_dset_path):
            self.log.warning("Featurized file already exists. Continuing:")
        else:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                #set_group_permissions(self.params.system, data_dir, self.params.data_owner, self.params.data_owner_group)

            featurized_dset_df.to_csv(featurized_dset_path, index=False)
            #set_group_permissions(self.params.system, featurized_dset_path, self.params.data_owner, self.params.data_owner_group)

    # ****************************************************************************************
    def load_featurized_data(self):
        """Loads prefeaturized data from the filesystem. Returns a data frame,
        which is then passed to featurization.extract_prefeaturized_data() for processing.

        Returns:
            featurized_dset_df (pd.DataFrame): dataframe of the prefeaturized data, needs futher processing
        """
        # First check to set if dataset already has the feature columns we need
        dset_df = self.load_full_dataset()
        if self.has_all_feature_columns(dset_df):
            self.dataset_key = self.params.dataset_key
            return dset_df


        # Otherwise, generate the expected path for the featurized dataset
        featurized_dset_name = self.featurization.get_featurized_dset_name(self.dataset_name)
        dataset_dir = os.path.dirname(self.params.dataset_key)
        data_dir = os.path.join(dataset_dir, self.featurization.get_featurized_data_subdir())
        featurized_dset_path = os.path.join(data_dir, featurized_dset_name)
        featurized_dset_df = pd.read_csv(featurized_dset_path)

        # check if featurized dset has all the smiles from dset_df
        dsetsmi=set(dset_df[self.params.smiles_col])
        featsmi=set(featurized_dset_df[self.params.smiles_col])
        if not dsetsmi-featsmi==set():
            raise AssertionError("All of the smiles in your dataset are not represented in your featurized file. You can set previously_featurized to False and your featurized dataset located in the scaled_descriptors directory will be overwritten to include the correct data.")
        
        self.dataset_key = featurized_dset_path
        featurized_dset_df[self.params.id_col] = featurized_dset_df[self.params.id_col].astype(str)

        

        return featurized_dset_df

    # ****************************************************************************************

    def save_split_dataset(self, directory=None):
        """Saves a table of compound IDs and split subset assignments for the current dataset.

        Args:
            directory (str): Directory where the split table will be created. Defaults to the directory
            of the current dataset.
        """

        split_df = self.create_dataset_split_table()
        if directory is None:
            directory = os.path.dirname(self.params.dataset_key)
        split_table_file = '{0}/{1}'.format(directory, self._get_split_key())
        split_df.to_csv(split_table_file, index=False)

        self.log.info('Dataset split table saved to %s' % split_table_file)

    # ****************************************************************************************
    def load_dataset_split_table(self, directory=None):
        """Loads from the filesystem a table of compound IDs assigned to each split subset of a dataset.
        Called by load_presplit_dataset().

        Args:
            directory (str): Directory where the split table will be created. Defaults to the directory
            of the current dataset.

        Returns:
            split_df (DataFrame): Table assigning compound IDs to split subsets and folds.
            split_kv: None for the FileDataset version of this method.
        """
        if directory is None:
            directory = os.path.dirname(self.params.dataset_key)
        split_table_file = '{0}/{1}'.format(directory, self._get_split_key())
        split_df = pd.read_csv(split_table_file, index_col=False)
        return split_df, None

class ClassificationDataException(Exception):
    """Used when dataset for classification problem violates assumptions
       -   Every subset in a split must have all classes
       -   Labels must range from 0 <= L < num_classes. DeepChem requires this.
           Errors occur when L > num_classes or L < 0
    """
    pass
