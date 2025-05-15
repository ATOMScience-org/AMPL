"""Classes providing different methods of transforming response data and/or features in datasets, beyond those
provided by DeepChem.
"""

import logging

import numpy as np
import umap

from deepchem.trans.transformers import Transformer, NormalizationTransformer, BalancingTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

logging.basicConfig(format='%(asctime)-15s %(message)s')
log = logging.getLogger('ATOM')

transformed_featurizers = ['descriptors', 'computed_descriptors']

# ****************************************************************************************
def transformers_needed(params):
    """Returns a boolean indicating whether response and/or feature transformers would be
    created for a model with the given parameters.

    Args:
        params (argparse.namespace: Object containing the parameter list

    Returns:
        boolean: True if transformers are required given the model parameters.
    """
    return ((params.featurizer in transformed_featurizers) or
           ((params.prediction_type == 'regression') and params.transformers))

# ****************************************************************************************
def get_statistics_missing_ydata(dataset):
    """Compute and return statistics of this dataset.

       This updated version gives the option to check for and ignore missing
       values for the y variable only. The x variable still assumes no
       missing values.
    """
    if len(dataset.y.shape)==1:
        num_tasks = 1
    else:
        num_tasks = dataset.y.shape[1]
    y_means = np.zeros(num_tasks)
    y_m2 = np.zeros(num_tasks)
    dy = np.zeros(num_tasks)
    n = np.zeros(num_tasks)
    for _, y, w, _ in dataset.itersamples():
       for it in range(len(y)) :
            ## set weights to 0 for missing data
            if np.isnan(y[it]) :
                assert(w[it]==0)
            if w[it]!=0 and not np.isnan(y[it]) :
               #print("huh",w[it],y[it])
               n[it]+=1
               dy[it] = y[it] - y_means[it]
               y_means[it] += dy[it] / n[it]
               y_m2[it] += dy[it] * (y[it] - y_means[it])

    y_stds=np.zeros(len(n))
    for it in range(len(n)) :
       if n[it] >= 2:
         y_stds[it] = np.sqrt(y_m2[it] / n[it])
    return y_means, y_stds

# ****************************************************************************************
def create_feature_transformers(params, featurization, train_dset):
    """Fit a scaling and centering transformation to the feature matrix of the given dataset, and return a
    DeepChem transformer object holding its parameters.

    Args:
        params (argparse.namespace): Object containing the parameter list

        featurization (featurization.Featurization): A Featurization object that will be used with
        the train_dset object.

        train_dset (dc.Dataset): Contains the dataset used to fit the the transformers.

    Returns:
        (list of DeepChem transformer objects): list of transformers for the feature matrix
    """
    if params.feature_transform_type == 'umap':
        # Map feature vectors using UMAP for dimension reduction
        log.warning("UMAP feature transformation is deprecated and will be removed in a future release.")
        if params.split_strategy == 'k_fold_cv':
            log.warning("Warning: UMAP transformation may produce misleading results when used with K-fold split strategy.")
        transformers_x = [UMAPTransformer(params, train_dset)]
    elif params.transformers:
        # TODO: Transformers on responses and features should be controlled only by parameters
        # response_transform_type and feature_transform_type, rather than params.transformers.

        # Scale and center feature matrix if featurization type calls for it
        transformers_x = featurization.create_feature_transformer(train_dset)
    else:
        transformers_x = []

    return transformers_x

# ****************************************************************************************
def create_weight_transformers(params, dataset):
    """Fit an optional balancing transformation to the weight matrix of the given dataset, and return a
    DeepChem transformer object holding its parameters.

    Args:
        params (argparse.namespace: Object containing the parameter list

        dataset (dc.Dataset): Contains the dataset to be transformed.

    Returns:
        (list of DeepChem transformer objects): list of transformers for the weight matrix
    """
    if params.weight_transform_type == 'balancing':
        if params.prediction_type == 'classification':
            transformers_w = [BalancingTransformer(dataset)]
        else:
            log.warning("Warning: Balancing transformer only supported for classification models.")
            transformers_w = []
    else:
        transformers_w = []

    return transformers_w

# ****************************************************************************************
def get_transformer_specific_metadata(params):
    """Returns a dictionary of parameters related to the currently selected transformer(s).

    Args:
        params (argparse.namespace: Object containing the parameter list

    Returns:
        meta_dict (dict): Nested dictionary of parameters and values for each currently active
        transformer.
    """
    meta_dict = {}
    if params.feature_transform_type == 'umap':
        umap_dict = dict(
                        umap_dim = params.umap_dim,
                        umap_metric = params.umap_metric,
                        umap_targ_wt = params.umap_targ_wt,
                        umap_neighbors = params.umap_neighbors,
                        umap_min_dist = params.umap_min_dist )
        meta_dict['umap_specific'] = umap_dict
    return meta_dict

# ****************************************************************************************
def get_transformer_keys(params):
    """Makes all transformer keys
    There is one set of transformers for each fold and then one transformer
    for both validation and training sets. AMPL automatically trains a model
    using all validation and training data at the end of the training loop.

    Args:
        params (argparse.namespace: Object containing the parameter list

    Returns:
        (list): A list of all keys used in transformer dictionaries.
    """
    if params.split_strategy != 'k_fold_cv':
        return [0, 'final']
    else:
        return list(range(params.num_folds))+['final']

# ****************************************************************************************
def get_blank_transformations():
    """Get empty transformations dictionary
    These keys must always exist, even when there are no transformations

    Returns:
        (dict): A dictionary containing empty lists. Used when no transformers are needed
    """
    return {0:[], 'final':[]}

# ****************************************************************************************
def get_all_training_datasets(model_dataset):
    """Returns all 'training' datasets
    This takes a model_dataset and returns a dictionary of all
    datasets that will need a transformer. The keys will match
    what is returned by get_transformer_keys

    Args:
        model_dataset: A model_datasets.ModelDataset object containing the current dataset.

    Returns:
        dict of dc.Datasets: A dictionary keyed using keys fold numbers and 'final'. Contains
        the training data for each fold and the final training+validation training set.
    """
    result = {}
    if model_dataset.splitting is None:
        # this dataset is not split into training and validation, use all data
        result['final'] = model_dataset.dataset
    elif len(model_dataset.train_valid_dsets)==1:
        # there is only one fold, use the training set from that
        # for random forests and xgboost models, the final and
        # 0th fold are the same if there k-fold is not used
        result['final'] = model_dataset.train_valid_dsets[0][0]
        result[0] = model_dataset.train_valid_dsets[0][0]
    else:
        # First, get the training set from all the folds
        for i, (t, v) in enumerate(model_dataset.train_valid_dsets):
            result[i] = t

        # Next, add the dataset that contains all training+validation data
        result['final'] = model_dataset.combined_training_data()

    return result


# ****************************************************************************************
class UMAPTransformer(Transformer):
    """Dimension reduction transformations using the UMAP algorithm.
    
    DEPRECATED: Will be removed in a future release.

    Attributes:
        mapper (UMAP) : UMAP transformer
        scaler (RobustScaler): Centering/scaling transformer

    """
    def __init__(self, params, dataset):
        """Initializes a UMAPTransformer object.

        Args:
            params (Namespace): Contains parameters used to instantiate the transformer.

            dataset (Dataset): Dataset used to "train" the projection mapping.
        """

        # TODO: decide whether to make n_epochs a parameter
        #default_n_epochs = None
        default_n_epochs = 500

        if params.prediction_type == 'classification':
            target_metric = 'categorical'
        else:
            target_metric = 'l2'
        self.scaler = RobustScaler()
        # Use SimpleImputer to replace missing values (NaNs) with means for each column
        self.imputer = SimpleImputer()
        scaled_X = self.scaler.fit_transform(self.imputer.fit_transform(dataset.X))
        self.mapper = umap.UMAP(n_neighbors=params.umap_neighbors,
                                n_components=params.umap_dim,
                                metric=params.umap_metric,
                                target_metric=target_metric,
                                target_weight=params.umap_targ_wt,
                                min_dist=params.umap_min_dist,
                                n_epochs=default_n_epochs)
        # TODO: How to deal with multitask data?
        self.mapper.fit(scaled_X, y=dataset.y.flatten())

    # ****************************************************************************************
    def transform(self, dataset, parallel=False):
        return super(UMAPTransformer, self).transform(dataset, parallel=parallel)

    # ****************************************************************************************
    def transform_array(self, X, y, w, ids):
        X = self.mapper.transform(self.scaler.transform(self.imputer.transform(X)))
        return (X, y, w, ids)

    # ****************************************************************************************
    def untransform(self, z):
        """Reverses stored transformation on provided data."""
        raise NotImplementedError("Can't reverse a UMAP transformation")
    # ****************************************************************************************


# ****************************************************************************************

class NormalizationTransformerMissingData(NormalizationTransformer):
    """Test extension to check for missing data"""
    def __init__(self,
                 transform_X=False,
                 transform_y=False,
                 transform_w=False,
                 dataset=None,
                 move_mean=True) :

        if transform_X :
            X_means, X_stds = dataset.get_statistics(X_stats=True, y_stats=False)
            self.X_means = X_means
            self.X_stds = X_stds
        elif transform_y:
            y_means, y_stds = get_statistics_missing_ydata(dataset)
            self.y_means = y_means
            # Control for pathological case with no variance.
            y_stds = np.array(y_stds)
            y_stds[y_stds == 0] = 1.
            self.y_stds = y_stds
            self.move_mean = move_mean

       ## skip the NormalizationTransformer initialization and go to base class
        super(NormalizationTransformer, self).__init__(
                transform_X=transform_X,
                transform_y=transform_y,
                transform_w=transform_w,
                dataset=dataset)

    def transform(self, dataset, parallel=False):
        return dataset.transform(self)

    def transform_array(self, X, y, w, ids):
        """Transform the data in a set of (X, y, w) arrays."""
        if self.transform_X:
            zero_std_pos = np.where(self.X_stds == 0)
            X_weight = np.ones_like(self.X_stds)
            X_weight[zero_std_pos] = 0
            if not hasattr(self, 'move_mean') or self.move_mean:
                X = np.nan_to_num((X - self.X_means) * X_weight / self.X_stds)
            else:
                X = np.nan_to_num(X * X_weight / self.X_stds)
            # zero out large values, especially for out of range test data
            X[np.abs(X) > 1e30] = 1e30
        if self.transform_y:
            if not hasattr(self, 'move_mean') or self.move_mean:
                y = np.nan_to_num((y - self.y_means) / self.y_stds)
            else:
                y = np.nan_to_num(y / self.y_stds)
        return (X, y, w, ids)

    def untransform(self, z: np.ndarray) -> np.ndarray:
        """Undo transformation on provided data.

        Overrides DeepChem NormalizationTransformer method to fix issue #1821.

        Parameters
        ----------
        z: np.ndarray
            Array to transform back

        Returns
        -------
        z_out: np.ndarray
            Array with normalization undone.
        """
        if self.transform_X:
            if not hasattr(self, 'move_mean') or self.move_mean:
                return z * self.X_stds + self.X_means
            else:
                return z * self.X_stds
        elif self.transform_y:
            y_stds = self.y_stds
            y_means = self.y_means
            z_shape = list(z.shape)
            # Get the reversed shape of z: (..., n_tasks, batch_size)
            z_shape.reverse()
            # Find the task dimension of z
            for ind, dim in enumerate(z_shape):
                if ind < (len(z_shape) - 1) and dim == 1:
                    # Prevent broadcasting on wrong dimension
                    y_stds = np.expand_dims(y_stds, -1)
                    y_means = np.expand_dims(y_means, -1)
            if not hasattr(self, 'move_mean') or self.move_mean:
                return z * y_stds + y_means
            else:
                return z * y_stds
        else:
            return z

# ****************************************************************************************

class NormalizationTransformerHybrid(NormalizationTransformer):
    """Test extension to check for missing data"""
    def __init__(self,
                 transform_X=False,
                 transform_y=False,
                 transform_w=False,
                 dataset=None,
                 move_mean=True) :

        if transform_X :
            X_means, X_stds = dataset.get_statistics(X_stats=True, y_stats=False)
            self.X_means = X_means
            self.X_stds = X_stds
        elif transform_y:
            ki_pos = np.where(np.isnan(dataset.y[:,1]))[0]
            bind_pos = np.where(~np.isnan(dataset.y[:,1]))[0]
            y_means = dataset.y[ki_pos, 0].mean()
            y_stds = dataset.y[ki_pos, 0].std()
            self.y_means = y_means
            # Control for pathological case with no variance.
            self.y_stds = y_stds
            self.move_mean = move_mean
            self.dataset = dataset
            # check the single dose data range
            y_mean_bind = dataset.y[bind_pos, 0].mean()
            if y_mean_bind > 2:
                raise Exception("The single-dose values have a mean value over 2, they are probably NOT in the fraction format, but a percentage format. Make sure the single-dose values are in fraction format.")
        self.ishybrid = True # used to distinguish this special transformer.

       ## skip the NormalizationTransformer initialization and go to base class
        super(NormalizationTransformer, self).__init__(
                transform_X=transform_X,
                transform_y=transform_y,
                transform_w=transform_w,
                dataset=dataset)

    def transform(self, dataset, parallel=False):
        return dataset.transform(self)

    def transform_array(self, X, y, w, ids):
        """Transform the data in a set of (X, y, w) arrays."""
        if self.transform_X:
            zero_std_pos = np.where(self.X_stds == 0)
            X_weight = np.ones_like(self.X_stds)
            X_weight[zero_std_pos] = 0
            if not hasattr(self, 'move_mean') or self.move_mean:
                X = np.nan_to_num((X - self.X_means) * X_weight / self.X_stds)
            else:
                X = np.nan_to_num(X * X_weight / self.X_stds)
        if self.transform_y:
            ki_pos = np.where(np.isnan(y[:,1]))[0]
            bind_pos = np.where(~np.isnan(y[:,1]))[0]
            if not hasattr(self, 'move_mean') or self.move_mean:
                y[ki_pos, 0] = (y[ki_pos, 0] - self.y_means) / self.y_stds
                y[bind_pos, 0] = np.minimum(0.999, np.maximum(0.001, y[bind_pos, 0]))
            else:
                y[ki_pos, 0] = y[ki_pos, 0] / self.y_stds
                y[bind_pos, 0] = np.minimum(0.999, np.maximum(0.001, y[bind_pos, 0]))
        return (X, y, w, ids)

    def untransform(self, z, isreal=True):
        if self.transform_X:
            if not hasattr(self, 'move_mean') or self.move_mean:
                return z * self.X_stds + self.X_means
            else:
                return z * self.X_stds
        elif self.transform_y:
            y_stds = self.y_stds
            y_means = self.y_means
            y = z.copy()
            if len(z.shape) > 1 and z.shape[1] > 1:
                ki_pos = np.where(np.isnan(z[:,1]))[0]
                bind_pos = np.where(~np.isnan(z[:,1]))[0]
                y[ki_pos, 0] = z[ki_pos, 0] * y_stds + y_means
                if isreal:
                    y[bind_pos, 0] = z[bind_pos, 0]
                else:
                    # note that in prediction, all posistions are predicted as pKi, then bind positions get converted.
                    y[bind_pos, 0] = z[bind_pos, 0] * y_stds + y_means
            else:
                # no conc column, treat all rows as ki/IC50
                y[:, 0] = z[:, 0] * y_stds + y_means

            return y
