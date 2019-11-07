"""
Classes providing different methods of transforming response data and/or features in datasets, beyond those
provided by DeepChem.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import umap

import deepchem as dc
from deepchem.trans.transformers import Transformer, NormalizationTransformer
from sklearn.preprocessing import RobustScaler, Imputer

logging.basicConfig(format='%(asctime)-15s %(message)s')
log = logging.getLogger('ATOM')

def get_statistics_missing_ydata(dataset):
    """Compute and return statistics of this dataset.

       This updated version gives the option to check for and ignore missing
       values for the y variable only. The x variable still assumes no
       missing values.
    """
    y_means = np.zeros(len(dataset.get_task_names()))
    y_m2 = np.zeros(len(dataset.get_task_names()))
    dy = np.zeros(len(dataset.get_task_names()))
    n = np.zeros(len(dataset.get_task_names()))
    for _, y, w, _ in dataset.itersamples():
       for it in range(len(y)) :
            ## set weights to 0 for missing data
            if np.isnan(y[it]) :
                assert(w[i]==0)
            if w[it]!=0 and not np.isnan(y[it]) :
               #print("huh",w[it],y[it]) 
               n[it]+=1
               dy[it] = y[it] - y_means[it]
               y_means[it] += dy[it] / n[it]
               y_m2[it] += dy[it] * (y[it] - y_means[it])
      
    print("n_cnt",n)
    print("y_means",y_means)
    y_stds=np.zeros(len(n))
    for it in range(len(n)) :
       if n[it] >= 2:
         y_stds[it] = np.sqrt(y_m2[it] / n[it])
    print("y_stds",y_stds)
    return y_means, y_stds
    
# ****************************************************************************************
def create_feature_transformers(params, model_dataset):
    """Fit a scaling and centering transformation to the feature matrix of the given dataset, and return a
    DeepChem transformer object holding its parameters.

    Args:
        params (argparse.namespace: Object containing the parameter list
        model_dataset (ModelDataset): Contains the dataset to be transformed.
    Returns:
        (list of DeepChem transformer objects): list of transformers for the feature matrix
    """
    if params.feature_transform_type == 'umap':
        # Map feature vectors using UMAP for dimension reduction
        if model_dataset.split_strategy == 'k_fold_cv':
            log.warning("Warning: UMAP transformation may produce misleading results when used with K-fold split strategy.")
        train_dset = model_dataset.train_valid_dsets[0][0]
        transformers_x = [UMAPTransformer(params, train_dset)]
    elif params.transformers==True:
        # TODO: Transformers on responses and features should be controlled only by parameters
        # response_transform_type and feature_transform_type, rather than params.transformers.

        # Scale and center feature matrix if featurization type calls for it
        transformers_x = model_dataset.featurization.create_feature_transformer(model_dataset.dataset)
    else:
        transformers_x = []

    return transformers_x

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
        meta_dict['UmapSpecific'] = umap_dict
    return meta_dict

# ****************************************************************************************

class UMAPTransformer(Transformer):
    """
    Dimension reduction transformations using the UMAP algorithm.

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
        # Use Imputer to replace missing values (NaNs) with means for each column
        self.imputer = Imputer()
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
    def transform_array(self, X, y, w):
        X = self.mapper.transform(self.scaler.transform(self.imputer.transform(X)))
        return (X, y, w)

    # ****************************************************************************************
    def untransform(self, z):
        """Reverses stored transformation on provided data."""
        raise NotImplementedError("Can't reverse a UMAP transformation")
    # ****************************************************************************************

    
# ****************************************************************************************

class NormalizationTransformerMissingData(NormalizationTransformer):
    """
    Test extension to check for missing data
    """
    def __init__(self,
                 transform_X=False,
                 transform_y=False,
                 transform_w=False,
                 dataset=None,
                 transform_gradients=False,
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
            self.transform_gradients = transform_gradients
            self.move_mean = move_mean
            if self.transform_gradients:
              true_grad, ydely_means = get_grad_statistics(dataset)
              self.grad = np.reshape(true_grad, (true_grad.shape[0], -1, 3))
              self.ydely_means = ydely_means

       ## skip the NormalizationTransformer initialization and go to base class
       super(NormalizationTransformer, self).__init__(
                transform_X=transform_X,
                transform_y=transform_y,
                transform_w=transform_w,
                dataset=dataset)


