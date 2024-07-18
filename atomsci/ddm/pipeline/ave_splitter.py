"""Code to split a DeepChem dataset in such a way as to minimize the AVE bias, as described in `this paper by Wallach & Heifets
<https://pubs.acs.org/doi/10.1021/acs.jcim.7b00403>`_

Although the AVEMinSplitter class and its methods are public, you will typically not call them directly. Instead, they are invoked by
setting `splitter` to 'ave_min' in the model parameters when you train a model.
"""

# Portions of the code below are Copyright 2017 Atomwise Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from deepchem.splits.splitters import Splitter
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import pandas as pd
from numpy.random import shuffle, permutation
import random
from multiprocessing import Pool

from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import featurization as feat
from atomsci.ddm.pipeline import model_datasets as md

import logging
logging.basicConfig(format='%(asctime)-15s %(message)s')
# Set up logging
log = logging.getLogger('ATOM')


POP_SIZE = 100
NEXT_GEN_FACTOR = 20
MAX_SET_OVERLAP_FRACTION = 0.8
RM_PROB = 0.2
ADD_PROB = 0.2
MIN_AV = 10
MIN_IV = 10
MIN_AT = 30
MIN_IT = 30
TARGET_BIAS = 0.01


#*******************************************************************************************************************************************
def _Cdist(params):
    """
    Single-argument wrapper for scipy.spatial.distance.cdist, to be called from Pool.map()

    Args:
        params (list): Three element list containing the two feature arrays and the name of the metric to be used
            to compute their distances

    Returns:
        np.ndarray: Compressed distance matrix

    """
    return cdist(params[0], params[1], params[2]) 

#*******************************************************************************************************************************************
def _calc_dist_mat(feat1, feat2, metric, pool, num_workers):
    """Calculate distance matrix between rows of feat1 and rows of feat2

    Args:
        feat1 (np.ndarray): First feature array

        feat2 (np.ndarray): Second feature array

        metric (str): Name of metric to use (e.g. 'jaccard', 'euclidean')

        pool (multiprocessing.Pool or None): Pool of workers to parallelize calculation

        num_workers (int) Number of parallel workers:

    Returns:
        The distance matrix.

    """
    if (num_workers > 1 ):
        interval, remainder = divmod( len(feat1), min( len(feat1), num_workers-1 ) )
        interval = max( interval, 1 )
        chunks = pool.map( _Cdist, [ (feat1[r:min(r+interval, len(feat1) ) ], feat2, metric) for r in range( 0, len(feat1), interval ) ] )
        dist_mat = np.vstack( chunks ) 
    else :
        dist_mat = cdist( feat1, feat2, metric )
    return dist_mat


#*******************************************************************************************************************************************
def analyze_split(params, id_col='compound_id', smiles_col='rdkit_smiles', active_col='active'):
    """Evaluate the AVE bias for the training/validation and training/test set splits of the given dataset.

    Also show the active frequencies in each subset and for the dataset as a whole.
    id_col, smiles_col and active_col are defaults to be used in case they aren't found in the dataset metadata; if found
    the metadata values are used instead.

    Args:
        params (argparse.Namespace): Pipeline parameters.

        id_col (str): Dataset column containing compound IDs.

        smiles_col (str): Dataset column containing SMILES strings.

        active_col (str): Dataset column containing binary classifications.

    Returns:
        :obj:`pandas.DataFrame`: Table of split subsets showing sizes, numbers and fractions of active compounds

    """
    dset_key = params.dataset_key
    bucket = params.bucket
    split_uuid = params.split_uuid

    ds_client = dsf.config_client()
    try:
        split_metadata = dsf.search_datasets_by_key_value('split_dataset_uuid', split_uuid, ds_client, operator='in', bucket=bucket)
        split_oid = split_metadata['dataset_oid'].values[0]
        split_df = dsf.retrieve_dataset_by_dataset_oid(split_oid, client=ds_client)
    except Exception as e:
        print("Error when loading split file:\n%s" % str(e))
        raise
    
    try:
        dataset_df = dsf.retrieve_dataset_by_datasetkey(dset_key, bucket, client=ds_client)
        dataset_meta = dsf.retrieve_dataset_by_datasetkey(dset_key, bucket, client=ds_client, return_metadata=True)
    except Exception as e:
        print("Error when loading dataset:\n%s" % str(e))
        raise
    kv_dict = dsf.get_key_val(dataset_meta['metadata'])
    id_col = kv_dict.get('id_col', id_col)
    smiles_col = kv_dict.get('smiles_col', smiles_col)
    active_col = kv_dict.get('response_col', active_col)

    try:
        print('Dataset has %d unique compound IDs' % len(set(dataset_df[id_col].values)))
        print('Split table has %d unique compound IDs' % len(set(split_df.cmpd_id.values)))

        dset_df = dataset_df.merge(split_df, how='inner', left_on=id_col, right_on='cmpd_id').drop('cmpd_id', axis=1)
    except Exception as e:
        print("Error when joining dataset with split dataset:\n%s" % str(e))
        raise

    featurization = feat.create_featurization(params)
    data = md.create_model_dataset(params, featurization, ds_client)
    data.get_featurized_data()
    feat_arr = data.dataset.X
    # TODO: impute missing values if necessary
    y = data.dataset.y.flatten()
    if len(set(y) - set([0,1])) > 0:
        raise ValueError('AVEMinSplitter only works on binary classification datasets')
    ids = data.dataset.ids
    active_ind = np.where(y == 1)[0]
    inactive_ind = np.where(y == 0)[0]
    active_feat = feat_arr[active_ind,:]
    inactive_feat = feat_arr[inactive_ind,:]
    num_active = len(active_ind)
    num_inactive = len(inactive_ind)
    active_ids = ids[active_ind]
    inactive_ids = ids[inactive_ind]
    active_id_ind = dict(zip(active_ids, range(len(active_ids))))
    inactive_id_ind = dict(zip(inactive_ids, range(len(inactive_ids))))
    if params.featurizer == 'ecfp':
        metric = 'jaccard'
    elif params.featurizer == 'graphconv':
        raise ValueError("ave_min splitter dopesn't support graphconv features")
    else:
        metric = 'euclidean'

    # Calculate distance thresholds where nearest neighborfunction should be evaluated
    if metric == 'jaccard':
        max_nn_dist = 1.0
    else:
        nan_mat = np.isnan(feat_arr)
        nnan = np.sum(nan_mat)
        if nnan > 0:
            log.info('Input feature matrix has %d NaN elements' % nnan)
            not_nan = ~nan_mat
            for i in range(feat_arr.shape[1]):
                feat_arr[nan_mat[:,i],i] = np.mean(feat_arr[not_nan[:,i],i])
        nn_dist = np.sort(squareform(pdist(feat_arr, metric)))[:,1]
        med_nn_dist = np.median(nn_dist)
        max_nn_dist = 3.0*med_nn_dist
    ndist = 100
    dist_thresh = np.linspace(0.0, max_nn_dist, ndist)

    # Compute distance matrices between subsets
    num_workers = 1
    aa_dist = _calc_dist_mat(active_feat, active_feat, metric, None, num_workers )
    ii_dist = _calc_dist_mat(inactive_feat, inactive_feat, metric, None, num_workers )
    ai_dist = _calc_dist_mat(active_feat, inactive_feat, metric, None, num_workers )
    ia_dist = ai_dist.transpose()

    subsets = sorted(set(dset_df.subset.values))
    subset_active_ind = {}
    subset_inactive_ind = {}

    if 'train' in subsets:
        # this is a TVT split
        subsets = ['train', 'valid', 'test']
        for subset in subsets:
            subset_df = dset_df[dset_df.subset == subset]
            active_df = subset_df[subset_df[active_col] == 1]
            inactive_df = subset_df[subset_df[active_col] == 0]
            subset_active_ids = active_df[id_col].values
            subset_inactive_ids = inactive_df[id_col].values
            subset_active_ind[subset] = [active_id_ind[id] for id in subset_active_ids]
            subset_inactive_ind[subset] = [inactive_id_ind[id] for id in subset_inactive_ids]

        taI = subset_active_ind['train']
        tiI = subset_inactive_ind['train']
        print("Results for %s split with %s %s features:" % (params.splitter, params.descriptor_type, params.featurizer))
        for valid_set in ['valid', 'test']:
            vaI = subset_active_ind[valid_set]
            viI = subset_inactive_ind[valid_set]
            split_params = ((vaI, viI, taI, tiI), aa_dist, ii_dist, ai_dist, ia_dist, dist_thresh)
            _plot_nn_dist_distr(split_params)
            bias = _plot_bias(split_params, niter=0)
            print("For train/%s split: AVE bias = %.5f" % (valid_set, bias))
    else:
        # TODO: deal with k-fold splits later
        print('k-fold CV splits not supported yet')
        return

    # Tabulate the fractions of actives in the full dataset and each subset
    subset_list = []
    size_list = []
    frac_list = []
    active_frac_list = []

    dset_size = data.dataset.X.shape[0]
    dset_active = sum(data.dataset.y)
    subset_list.append('full dataset')
    size_list.append(dset_size)
    frac_list.append(1.0)
    active_frac_list.append(dset_active/dset_size)

    for subset in subsets:
        active_size = len(subset_active_ind[subset])
        inactive_size = len(subset_inactive_ind[subset])
        subset_size = active_size+inactive_size
        active_frac = active_size/subset_size
        subset_list.append(subset)
        size_list.append(subset_size)
        frac_list.append(subset_size/dset_size)
        active_frac_list.append(active_frac)
    frac_df = pd.DataFrame(dict(subset=subset_list, size=size_list, fraction=frac_list, active_frac=active_frac_list))
    print('\nSplit subsets:')
    print(frac_df)

    return frac_df


#*******************************************************************************************************************************************
def _check_split_similarity( params ):
    """Compare index sets given by params[0:4] against the corresponding sets given by params[4] and return True if
    any have more than a certain fraction of molecules in common.

    Args:
        params (tuple): Tuple of four index sets and a list of comparator index sets

    Returns:
        bool: True if any of the index sets has more than MAX_SET_OVERLAP_FRACTION of its compounds in common
            with the corresponding comparator set.
    """
    bias_split = params[4]
    for i in range(4):
        set1 = params[i]
        set2 = set(bias_split[1][i])
        max_overlap = MAX_SET_OVERLAP_FRACTION * len(set1)
        if len(set1 & set2) > max_overlap:
         return True
    return False

#*******************************************************************************************************************************************
def _calc_bias(params):
    """Compute the AVE bias objective function for the split set given by params[0], based on the distance matrices in the remaining params

    Args:
        params (tuple): Split index sets, distance matrices and thresholds

    Returns:
        float: The AVE bias for the given split

    """
    split_set, aa_dist, ii_dist, ai_dist, ia_dist, thresholds = params[:]
    vaI, viI, taI, tiI = split_set
    
    # get the slices of the distance matrices
    aTest_aTrain_D = aa_dist[ np.ix_( vaI, taI ) ]
    aTest_iTrain_D = ai_dist[ np.ix_( vaI, tiI ) ]
    iTest_aTrain_D = ia_dist[ np.ix_( viI, taI ) ] 
    iTest_iTrain_D = ii_dist[ np.ix_( viI, tiI ) ]

    # Compute the nearest training set neighbor distances for each test set compound 
    aa_nn_dist = np.min(aTest_aTrain_D, axis=1)
    ai_nn_dist = np.min(aTest_iTrain_D, axis=1)
    ia_nn_dist = np.min(iTest_aTrain_D, axis=1)
    ii_nn_dist = np.min(iTest_iTrain_D, axis=1)
 
    aTest_aTrain_func = [ np.mean(aa_nn_dist < t) for t in thresholds ]
    aTest_iTrain_func = [ np.mean(ai_nn_dist < t) for t in thresholds ]
    iTest_aTrain_func = [ np.mean(ia_nn_dist < t) for t in thresholds ]
    iTest_iTrain_func = [ np.mean(ii_nn_dist < t) for t in thresholds ]

    aTest_aTrain_S = np.mean(aTest_aTrain_func)
    aTest_iTrain_S = np.mean(aTest_iTrain_func)
    iTest_iTrain_S = np.mean(iTest_iTrain_func)
    iTest_aTrain_S = np.mean(iTest_aTrain_func)

    # Make debiasing more stringent by preventing negative and positive components from cancelling each other out
    aBias = abs(aTest_aTrain_S - aTest_iTrain_S)
    iBias = abs(iTest_iTrain_S - iTest_aTrain_S)
    bias = aBias + iBias
    return bias


#*******************************************************************************************************************************************
def _plot_nn_dist_distr(params):
    """Plot distributions of nearest neighbor distances

    Args:
        params (tuple): Split index sets, distance matrices and thresholds

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    split_set, aa_dist, ii_dist, ai_dist, ia_dist, thresholds = params[:]
    vaI, viI, taI, tiI = split_set
    
    # get the slices of the distance matrices
    aTest_aTrain_D = aa_dist[ np.ix_( vaI, taI ) ]
    aTest_iTrain_D = ai_dist[ np.ix_( vaI, tiI ) ]
    iTest_aTrain_D = ia_dist[ np.ix_( viI, taI ) ] 
    iTest_iTrain_D = ii_dist[ np.ix_( viI, tiI ) ]

    aa_nn_dist = np.min(aTest_aTrain_D, axis=1)
    ai_nn_dist = np.min(aTest_iTrain_D, axis=1)
    ia_nn_dist = np.min(iTest_aTrain_D, axis=1)
    ii_nn_dist = np.min(iTest_iTrain_D, axis=1)

    # Plot distributions of nearest-neighbor distances
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    sns.kdeplot(aa_nn_dist, ax=axes[0,0])
    axes[0,0].set_title('AA')
    sns.kdeplot(ai_nn_dist, ax=axes[0,1])
    axes[0,1].set_title('AI')
    sns.kdeplot(ia_nn_dist, ax=axes[1,0])
    axes[1,0].set_title('II')
    sns.kdeplot(ii_nn_dist, ax=axes[1,1])
    axes[1,1].set_title('IA')

#*******************************************************************************************************************************************
def _plot_bias(params, niter):
    """Plot nearest neighbor functions used to compute AVE bias. Used in place of _calc_bias for debugging splitter code.

    Args:
        params (tuple): Split index sets, distance matrices and thresholds

    Returns:
        float: The AVE bias for the given split.

    """
    import matplotlib.pyplot as plt

    split_set, aa_dist, ii_dist, ai_dist, ia_dist, thresholds = params[:]
    vaI, viI, taI, tiI = split_set
    
    # get the slices of the distance matrices
    aTest_aTrain_D = aa_dist[ np.ix_( vaI, taI ) ]
    aTest_iTrain_D = ai_dist[ np.ix_( vaI, tiI ) ]
    iTest_aTrain_D = ia_dist[ np.ix_( viI, taI ) ] 
    iTest_iTrain_D = ii_dist[ np.ix_( viI, tiI ) ]

    aa_nn_dist = np.min(aTest_aTrain_D, axis=1)
    ai_nn_dist = np.min(aTest_iTrain_D, axis=1)
    ia_nn_dist = np.min(iTest_aTrain_D, axis=1)
    ii_nn_dist = np.min(iTest_iTrain_D, axis=1)

    aTest_aTrain_func = [ np.mean(aa_nn_dist < t) for t in thresholds ]
    aTest_iTrain_func = [ np.mean(ai_nn_dist < t) for t in thresholds ]
    iTest_aTrain_func = [ np.mean(ia_nn_dist < t) for t in thresholds ]
    iTest_iTrain_func = [ np.mean(ii_nn_dist < t) for t in thresholds ]

    aTest_aTrain_S = np.mean(aTest_aTrain_func)
    aTest_iTrain_S = np.mean(aTest_iTrain_func)
    iTest_iTrain_S = np.mean(iTest_iTrain_func)
    iTest_aTrain_S = np.mean(iTest_aTrain_func)

    #aBias = aTest_aTrain_S - aTest_iTrain_S
    #iBias = iTest_iTrain_S - iTest_aTrain_S
    #bias = aBias + iBias

    # ksm: Make debiasing more stringent by preventing negative and positive components from cancelling each other out
    aBias = abs(aTest_aTrain_S - aTest_iTrain_S)
    iBias = abs(iTest_iTrain_S - iTest_aTrain_S)
    bias = aBias + iBias

    fig, axes = plt.subplots(1, 2, figsize=(18,10))
    ax = axes[0]
    ax.plot(thresholds, aTest_aTrain_func, color='blue', label='AA')
    ax.plot(thresholds, aTest_iTrain_func, color='red', label='AI')
    ax.fill_between(thresholds, aTest_iTrain_func, aTest_aTrain_func, facecolor='blue', alpha=0.3, linewidth=0)
    ax.set_xlabel('Distance')
    ax.set_ylabel('NN Function')
    legend = ax.legend()
    title = "AA - AI = %.3f\nIteration: %d" % (aBias, niter)
    ax.set_title(title)

    ax = axes[1]
    ax.plot(thresholds, iTest_iTrain_func, color='blue', label='II')
    ax.plot(thresholds, iTest_aTrain_func, color='red', label='IA')
    ax.fill_between(thresholds, iTest_iTrain_func, iTest_aTrain_func, facecolor='hotpink', alpha=0.3, linewidth=0)
    ax.set_xlabel('Distance')
    ax.set_ylabel('NN Function')
    legend = ax.legend()
    title = "II - IA = %.3f\nTotal bias = %.3f" % (iBias, bias)
    ax.set_title(title)

    plt.show()


    return bias

#*******************************************************************************************************************************************
class AVEMinSplitter(Splitter):
    """Class for splitting a DeepChem dataset in order to minimize the Asymmetric Validation Embedding bias.

    Uses distances between feature vectors and binary classifications to compute
    the AVE bias for a candidate split and find a split that minimizes the bias.

    Attributes:
        metric (str): Name of the metric to be used to compute distances between feature vectors.

        verbose (bool): Ignored.

        num_workers (int): Number of threads to use to parallelize computations.

        max_iter (int): Maximmum number of iterations to execute to try to minimize bias.

        ndist (int): Number of points to use to approximate CDF of distance distribution.

        debug_mode (bool): If true, generate extra plots and log messages for debugging.

    """

    def __init__(self, metric='jaccard', verbose=True, num_workers=1, max_iter=300, ndist=100, debug_mode=False):

        self.verbose = verbose
        self.metric = metric
        self.num_workers = num_workers
        self.debug_mode = debug_mode
        self.max_iter = max_iter
        self.dist_thresh = None
        self.ndist = ndist

        # Set up multiprocessing pools
        if num_workers > 1:
            self.calc_dist_pool = Pool(processes = num_workers)
            self.split_sim_pool = Pool(processes = num_workers)
            self.calc_bias_pool = Pool(processes = num_workers)
        else:
            self.calc_dist_pool = None
            self.split_sim_pool = None
            self.calc_bias_pool = None

    def split(self, dataset, frac_train=0.8, frac_valid=0.2, frac_test=0.0, seed=None, log_every_n=None):
        """Split dataset into training and validation sets that minimize the AVE bias. A test set is not generated;
        to do a 3-way split, call this function twice.

        Args:
            dataset (dc.Dataset): The DeepChem dataset to be split

            frac_train (float): The approximate fraction of compounds to put in the training set

            frac_valid (float): The approximate fraction of compounds to put in the validation or test set

            frac_test (float): Ignored; included only for compatibility with the DeepChem Splitter API

            seed (int): Ignored

            log_every_n (int or None): Ignored

        Returns:
            tuple: Lists of indices of compounds assigned to the training and validation/test sets.

            The third element of the tuple is an empty list, because this function only does a 2-way split.

        Todo:
            Change code to do a 3-way split in one call, rather than requiring the distance matrices to be computed twice.
        """
        if self.debug_mode:
            import matplotlib.pyplot as plt
            import seaborn as sns

        feat = dataset.X
        # Compute overall nearest neighbor distances for each compound
        if self.metric == 'jaccard':
            max_nn_dist = 1.0
        else:
            nan_mat = np.isnan(feat)
            nnan = np.sum(nan_mat)
            if nnan > 0:
                log.info('Input feature matrix has %d NaN elements' % nnan)
                not_nan = ~nan_mat
                for i in range(feat.shape[1]):
                    feat[nan_mat[:,i],i] = np.mean(feat[not_nan[:,i],i])
            nn_dist = np.sort(squareform(pdist(feat, self.metric)))[:,1]
            med_nn_dist = np.median(nn_dist)
            max_nn_dist = 3.0*med_nn_dist
            if self.debug_mode:
                log.debug("Median NN distance = %.3f" % med_nn_dist)
                # Plot distribution of overall NN distances
                fig, ax = plt.subplots(figsize=(8,8))
                sns.kdeplot(nn_dist, ax=ax)
                ax.set_title('Overall NN distance distribution')
                plt.axvline(med_nn_dist, color='forestgreen', linestyle='--')
                plt.axvline(3.0*med_nn_dist, color='red', linestyle='--')
                plt.show()
        self.dist_thresh = np.linspace(0.0, max_nn_dist, self.ndist)

        y_dim = dataset.y.shape
        if len(y_dim) > 1 and y_dim[1] > 1:
            raise ValueError('AVEMinSplitter only works for single task datasets')
        y = dataset.y.flatten()
        if len(set(y) - set([0,1])) > 0:
            raise ValueError('AVEMinSplitter only works on binary classification datasets')
        active_ind = np.where(y == 1)[0]
        inactive_ind = np.where(y == 0)[0]
        active_feat = feat[active_ind,:]
        inactive_feat = feat[inactive_ind,:]
        num_active = len(active_ind)
        num_inactive = len(inactive_ind)

        aa_dist = _calc_dist_mat(active_feat, active_feat, self.metric, self.calc_dist_pool, self.num_workers )
        ii_dist = _calc_dist_mat(inactive_feat, inactive_feat, self.metric, self.calc_dist_pool, self.num_workers )
        ai_dist = _calc_dist_mat(active_feat, inactive_feat, self.metric, self.calc_dist_pool, self.num_workers )
        ia_dist = ai_dist.transpose()

        pop = []

        num_train_actives = int(num_active * frac_train)
        num_train_inactives = int(num_inactive * frac_train)
        assert(num_train_actives > 0 and num_train_inactives > 0 )

        # Map indices of active and inactive compounds to indices within the active and inactive sets
        active_arr = list(range(num_active))
        inactive_arr = list(range(num_inactive))

        # randomly select an initial population of splits
        while (len (pop ) < POP_SIZE ):
            shuffle(active_arr)
            shuffle(inactive_arr)
            # Each split set is a tuple: (valid_actives, valid_inactives, train_actives, train_inactives)
            pop.append((active_arr[num_train_actives:], inactive_arr[num_train_inactives:], 
                        active_arr[:num_train_actives], inactive_arr[:num_train_inactives]))

        if self.debug_mode:
            _plot_nn_dist_distr((pop[0], aa_dist, ii_dist, ai_dist, ia_dist, self.dist_thresh))
        for iter_count in range(self.max_iter):
            # Calculate biases for each split and remove splits that are similar to less biased splits
            log.debug("Calculating biases")
            if self.num_workers > 1:
                biases = self.calc_bias_pool.map(_calc_bias, ((split, aa_dist, ii_dist, ai_dist, ia_dist, self.dist_thresh) for split in pop))
            else:
                if self.debug_mode:
                    biases = []
                    for k, split in enumerate(pop):
                        if k == 0:
                            bias = _plot_bias((split, aa_dist, ii_dist, ai_dist, ia_dist, self.dist_thresh), iter_count)
                        else:
                            bias = _calc_bias((split, aa_dist, ii_dist, ai_dist, ia_dist, self.dist_thresh))
                        biases.append(bias)
                else:
                    biases = [_calc_bias((split, aa_dist, ii_dist, ai_dist, ia_dist, self.dist_thresh)) for split in pop]
            bias_splits = sorted(zip(biases, pop))
            num_splits = len(bias_splits)

            log.debug("Removing similar splits")
            skip_indices = set()
            for i in range(num_splits):
                if i in skip_indices:
                    continue
                active_valid = set(bias_splits[i][1][0])
                inactive_valid = set(bias_splits[i][1][1])
                active_train = set(bias_splits[i][1][2])
                inactive_train = set(bias_splits[i][1][3])

                indices_to_check = sorted(set(range(i+1, num_splits)) - skip_indices)
                if self.num_workers > 1:
                    results = self.split_sim_pool.map(_check_split_similarity, ((active_valid, inactive_valid, active_train, inactive_train, bias_splits[j])
                                                                for j in indices_to_check))
                else:
                    results = [_check_split_similarity((active_valid, inactive_valid, active_train, inactive_train, bias_splits[j]))
                                                                for j in indices_to_check]
                for j, is_sim in zip(indices_to_check, results):
                    if is_sim:
                        skip_indices.add(j)
                num_skipped = len(skip_indices)
                if num_splits - num_skipped < NEXT_GEN_FACTOR:
                    break
            # remove the top overlapping splits
            log.debug("Found %d splits with > 0.8 overlap with better scoring sets" % num_skipped)
            num_remove = min(num_skipped, num_splits - NEXT_GEN_FACTOR)
            # code added in case 0 removed
            remove_indices = sorted(skip_indices, reverse=True)[:num_remove]
            for i in remove_indices:
                del bias_splits[i] 
    
            # select the top NEXT_GEN_FACTOR sets
            log.debug("population size after similarity filter: %d" % len(bias_splits))
            log.debug("select the next generation")
            new_splits = bias_splits[ :NEXT_GEN_FACTOR ]
            best_split = bias_splits[ 0 ]
            best_bias = best_split[ 0 ]
   
            log.debug("iter = %d  best_bias = %.3f" % (iter_count, best_bias))
     
            # If bias for best split is less than our target, exit the genetic optimization loop
            if best_bias < TARGET_BIAS:
                 break
    
            # "Recombine" and "mutate" the top scoring (least biased) splits to make a new population of splits 
            log.debug("breed")
            pop = []
            while(len(pop ) < POP_SIZE ):
                # randomly choose a pair
                pair = random.sample(new_splits, 2 )
    
                # for each subset of indices, select a new set from the union of indices for that subset
                newActiveIndicesV = list(pair[0][1][0]) + list(pair[1][1][0]) 
                newInactiveIndicesV = list(pair[0][1][1]) + list(pair[1][1][1]) 
                newActiveIndicesT = list(pair[0][1][2]) + list(pair[1][1][2])
                newInactiveIndicesT = list(pair[0][1][3]) + list(pair[1][1][3]) 
  
                avSize = int(len(newActiveIndicesV )/2 ) 
                ivSize = int(len(newInactiveIndicesV )/2 ) 
                atSize = int(len(newActiveIndicesT )/2 ) 
                itSize = int(len(newInactiveIndicesT )/2 ) 
      
                newActiveIndicesV = np.unique(newActiveIndicesV )
                newInactiveIndicesV = np.unique(newInactiveIndicesV )
                newActiveIndicesT = np.unique(newActiveIndicesT )
                newInactiveIndicesT = np.unique(newInactiveIndicesT )
      
                # make sure there are no overlapping molecules between training/validation sets
                newActiveIndices = list(np.hstack((newActiveIndicesV, newActiveIndicesT) ) )
                newInactiveIndices = list(np.hstack((newInactiveIndicesV, newInactiveIndicesT) ) )
                overlapActives = list(set([ x for x in newActiveIndices if newActiveIndices.count(x ) > 1 ] ) )
                overlapInactives = list(set([ x for x in newInactiveIndices if newInactiveIndices.count(x ) > 1 ] ) )
                shuffle(overlapActives )
                shuffle(overlapInactives )
                for idx, overlapA in enumerate(overlapActives ):
                    if (idx % 2 ):
                             newActiveIndicesV = np.delete(newActiveIndicesV, np.where(newActiveIndicesV == overlapA ) )
                    else :
                             newActiveIndicesT = np.delete(newActiveIndicesT, np.where(newActiveIndicesT == overlapA ) )
                for idx, overlapI in enumerate(overlapInactives ):
                    if (idx % 2 ):
                             newInactiveIndicesV = np.delete(newInactiveIndicesV, np.where(newInactiveIndicesV == overlapI ) )
                    else :
                             newInactiveIndicesT = np.delete(newInactiveIndicesT, np.where(newInactiveIndicesT == overlapI ) )
  
                newActiveIndices = list(np.hstack((newActiveIndicesV, newActiveIndicesT) ) )
                newInactiveIndices = list(np.hstack((newInactiveIndicesV, newInactiveIndicesT) ) )
                assert(len(set([ x for x in newActiveIndices if newActiveIndices.count(x ) > 1 ] ) ) == 0 ) 
                assert(len(set([ x for x in newInactiveIndices if newInactiveIndices.count(x ) > 1 ] ) ) == 0 ) 
  
                avSize = min(avSize, len(newActiveIndicesV ) )
                ivSize = min(ivSize, len(newInactiveIndicesV ) )
                atSize = min(atSize, len(newActiveIndicesT ) )
                itSize = min(itSize, len(newInactiveIndicesT ) )
  
                if (np.random.rand() < RM_PROB and avSize > MIN_AV ):
                    avSize -= 1 
                if (np.random.rand() < ADD_PROB ):
                    avSize += 1
                if (np.random.rand() < RM_PROB and ivSize > MIN_IV ):
                    ivSize -= 1 
                if (np.random.rand() < ADD_PROB ):
                    ivSize += 1 
                if (np.random.rand() < RM_PROB and atSize > MIN_AT ):
                    atSize -= 1
                if (np.random.rand() < ADD_PROB ):
                    atSize += 1 
                if (np.random.rand() < RM_PROB and itSize > MIN_IT ):
                    itSize -= 1 
                if (np.random.rand() < ADD_PROB ):
                    itSize += 1 
      
                avSamp = random.sample(list(newActiveIndicesV), min(len(newActiveIndicesV ), max(avSize, MIN_AV ) ) ) 
                ivSamp = random.sample(list(newInactiveIndicesV), min(len(newInactiveIndicesV ), max(ivSize, MIN_IV ) ) )
                atSamp = random.sample(list(newActiveIndicesT), min(len(newActiveIndicesT ), max(atSize, MIN_AT ) ) )
                itSamp = random.sample(list(newInactiveIndicesT), min(len(newInactiveIndicesT ), max(itSize, MIN_IT ) ) )
   
                pop.append((avSamp, ivSamp, atSamp, itSamp)) 

        # End of genetic optimization loop
        if self.debug_mode:
            _plot_nn_dist_distr((best_split[1], aa_dist, ii_dist, ai_dist, ia_dist, self.dist_thresh))
            final_bias = _plot_bias((best_split[1], aa_dist, ii_dist, ai_dist, ia_dist, self.dist_thresh), iter_count)
        active_valid, inactive_valid, active_train, inactive_train = best_split[1]
        # Map indices within active/inactive sets back to indices in original dataset
        train_inds = permutation(np.concatenate((active_ind[active_train], inactive_ind[inactive_train])))
        valid_inds = permutation(np.concatenate((active_ind[active_valid], inactive_ind[inactive_valid])))
        log.debug("Final split has %d training, %d validation cmpds" % (len(train_inds), len(valid_inds)))
        return train_inds, valid_inds, []


