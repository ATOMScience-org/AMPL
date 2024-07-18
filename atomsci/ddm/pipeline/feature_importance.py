"""Functions to assess feature importance in AMPL models"""

import numpy as np
import pandas as pd
from collections import defaultdict


from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline.perf_data import negative_predictive_value

from deepchem.data.datasets import NumpyDataset

from sklearn import metrics
from scipy import stats
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

import matplotlib.pyplot as plt
import seaborn as sns

# The following import requires scikit-learn >= 0.23.1
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

import logging
logging.basicConfig(format='%(asctime)-15s %(message)s')


class _SklearnRegressorWrapper(BaseEstimator):
    """Class that implements the parts of the scikit-learn Estimator interface needed by the
    permutation importance code for AMPL regression models.
    """
    def __init__(self, model_pipeline):
        self.params = model_pipeline.params
        self.model = model_pipeline.model_wrapper.model

    def fit(self, dataset):
        return self.model.fit(dataset)

    def predict(self, X):
        dataset = NumpyDataset(X)
        y_pred = self.model.predict(dataset)
        return y_pred.reshape((-1, 1))

class _SklearnClassifierWrapper(BaseEstimator):
    """Class that implements the parts of the scikit-learn Estimator interface needed by the
    permutation importance code for AMPL classification models.
    """
    def __init__(self, model_pipeline):
        self.params = model_pipeline.params
        self.model = model_pipeline.model_wrapper.model
        # TODO: Change for > 2 classes
        self.classes_ = np.array([0,1], dtype='int')

    def fit(self, X, y):
        dataset = NumpyDataset(X, y=y)
        return self.model.fit(dataset)

    def predict(self, X):
        # change to return class labels
        dataset = NumpyDataset(X)
        probs = self.model.predict(dataset).reshape((-1,2))
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        dataset = NumpyDataset(X)
        probs = self.model.predict(dataset)
        return probs.reshape((-1,2))



def _get_estimator(model_pipeline):
    """Given an AMPL ModelPipeline object, returns an object that supports the scikit-learn estimator interface (in particular,
    the predict and predict_proba methods), for the purpose of running the permutation_importance function.

    Args:
        model_pipeline (ModelPipeline): AMPL model pipeline for a trained model

    Returns:
        estimator (sklearn.base.BaseEstimator): A scikit-learn Estimator object for the model.
    """
    pparams = model_pipeline.params
    wrapper = model_pipeline.model_wrapper
    if pparams.model_type == 'RF':
        # DeepChem model is a wrapper for an sklearn model, so return that
        return wrapper.model.model
    elif pparams.model_type == 'xgboost':
        # XGBoost model is wrapped by an sklearn model
        return wrapper.model.model
    elif pparams.model_type == 'hybrid':
        # TODO: Hybrid model requires special handling because of the two types of predictions
        raise ValueError("Hybrid models not supported yet")
    elif pparams.model_type == 'NN':
        # TODO: Find out if this branch will work for new DeepChem/PyTorch models (AttentiveFP, MPNN, etc.)
        if pparams.prediction_type == 'regression':
            return _SklearnRegressorWrapper(model_pipeline)
        else:
            return _SklearnClassifierWrapper(model_pipeline)
    else:
        raise ValueError(f"Unsupported model type {pparams.model_type}")

def _get_scorer(score_type):
    """Returns an sklearn.metrics.Scorer object that can be used to get model performance scores for
    various input feature sets.

    Args:
        score_type (str): Name of the scoring metric to use. This can be any of the standard values supported
        by sklearn.metrics.get_scorer; the AMPL-specific values 'npv', 'mcc', 'kappa', 'mae', 'rmse', 'ppv',
        'cross_entropy', 'bal_accuracy' and 'avg_precision' are also supported. Score types for which smaller
        values are better, such as 'mae', 'rmse' and 'cross_entropy' are mapped to their negative counterparts.

    Returns:
        scorer (callable): Function to compute scores for the given metric, such that greater scores are always better.
        This will have the signature `(estimator, X, y)`, where `estimator` is a model, `X` is the feature array and `y`
        is an array of ground truth labels.
    """
    # Handle the cases where the metric isn't implemented in scikit-learn, or is but doesn't have a predefined
    # label recognized by metrics.get_scorer
    if score_type == 'npv':
        return metrics.make_scorer(negative_predictive_value)
    elif score_type == 'mcc':
        return metrics.make_scorer(metrics.matthews_corrcoef)
    elif score_type == 'kappa':
        return metrics.make_scorer(metrics.cohen_kappa_score)

    # Otherwise, map the score types used in AMPL to the ones used in scikit-learn in the cases where they are different
    score_type_map = dict(
            mae = 'neg_mean_absolute_error',
            rmse = 'neg_root_mean_squared_error',
            ppv = 'precision',
            cross_entropy = 'neg_log_loss',
            bal_accuracy = 'balanced_accuracy',
            avg_precision = 'average_precision')
    sklearn_score_type = score_type_map.get(score_type, score_type)
    return metrics.get_scorer(sklearn_score_type)


# ===================================================================================================
def base_feature_importance(model_pipeline=None, params=None):
    """Minimal baseline feature importance function. Given an AMPL model (or the parameters to train a model),
    returns a data frame with a row for each feature. The columns of the data frame depend on the model type and
    prediction type. If the model is a binary classifier, the columns include  t-statistics and p-values
    for the differences between the means of the active and inactive compounds. If the model is a random forest,
    the columns will include the mean decrease in impurity (MDI) of each feature, computed by the scikit-learn
    feature_importances_ function. See the scikit-learn documentation for warnings about interpreting the MDI
    importance. For all models, the returned data frame will include feature names, means and standard deviations
    for each feature.

    This function has been tested on RFs and NNs with rdkit descriptors. Other models and feature combinations
    may not be supported.

    Args:
        model_pipeline (`ModelPipeline`): A pipeline object for a model that was trained in the current Python session
        or loaded from the model tracker or a tarball file. Either model_pipeline or params must be provided.

        params (`dict`): Parameter dictionary for a model to be trained and analyzed. Either model_pipeline or a
        params argument must be passed; if both are passed, params is ignored and the parameters from model_pipeline
        are used.

    Returns:
        (imp_df, model_pipeline, pparams) (tuple):
            imp_df (`DataFrame`): Table of feature importance metrics.
            model_pipeline (`ModelPipeline`): Pipeline object for model that was passed to or trained by function.
            pparams (`Namespace`): Parsed parameters of model.

    """
    log = logging.getLogger('ATOM')
    if model_pipeline is None:
        if params is None:
            raise ValueError("Either model_pipeline or params can be None but not both")
        # Train a model based on the parameters given
        pparams = parse.wrapper(params)
        model_pipeline = mp.ModelPipeline(pparams)
        model_pipeline.train_model()
    else:
        if params is not None:
            log.info("model_pipeline and params were both passed; ignoring params argument and using params from model")
        pparams = model_pipeline.params

    # Load the original training, validation and test data, if necessary
    try:
        model_data = model_pipeline.data
    except AttributeError:
        model_pipeline.featurization = model_pipeline.model_wrapper.featurization
        model_pipeline.load_featurize_data()
        model_data = model_pipeline.data

    # Get the list of feature column names
    #features = model_pipeline.model_wrapper.featurization.get_feature_columns()
    features = model_pipeline.featurization.get_feature_columns()
    nfeat = len(features)
    imp_df = pd.DataFrame({'feature': features})

    # Get the training, validation and test sets (we assume we're not using K-fold CV). These are DeepChem Dataset objects.
    (train_dset, valid_dset) = model_data.train_valid_dsets[0]
    test_dset = model_data.test_dset

    imp_df['mean_value'] = train_dset.X.mean(axis=0)
    imp_df['std_value'] = train_dset.X.std(axis=0)

    if pparams.prediction_type == 'classification':
        # Compute a t-statistic for each feature for the difference between its mean values for active and inactive compounds
        tstats = []
        pvalues = []
        active = train_dset.X[train_dset.y[:,0] == 1, :]
        inactive = train_dset.X[train_dset.y[:,0] == 0, :]

        log.debug("Computing t-statistics")
        for ifeat in range(nfeat):
            res = stats.ttest_ind(active[:,ifeat], inactive[:,ifeat], equal_var=True, nan_policy='omit')
            tstats.append(res.statistic)
            pvalues.append(res.pvalue)
        imp_df['t_statistic'] = tstats
        imp_df['ttest_pvalue'] = pvalues

    if pparams.model_type == 'RF':
        # Tabulate the MDI-based feature importances for random forest models
        # TODO: Does this work for XGBoost models too?
        rf_model = model_pipeline.model_wrapper.model.model
        imp_df['mdi_importance'] = rf_model.feature_importances_

    return imp_df, model_pipeline, pparams

# ===================================================================================================
def permutation_feature_importance(model_pipeline=None, params=None, score_type=None, nreps=60, nworkers=1,
                                   result_file=None):
    """Assess the importance of each feature used by a trained model by permuting the values of each feature in succession
    in the training, validation and test sets, making predictions, computing performance metrics, and measuring the effect
    of scrambling each feature on a particular metric.

    Args:
        model_pipeline (`ModelPipeline`): A pipeline object for a model that was trained in the current Python session
        or loaded from the model tracker or a tarball file. Either `model_pipeline` or `params` must be provided.

        params (`dict`): Parameter dictionary for a model to be trained and analyzed. Either `model_pipeline` or a
        `params` argument must be passed; if both are passed, `params` is ignored and the parameters from `model_pipeline`
        are used.

        score_type (str): Name of the scoring metric to use to assess importance. This can be any of the standard values
        supported by sklearn.metrics.get_scorer; the AMPL-specific values 'npv', 'mcc', 'kappa', 'mae', 'rmse', 'ppv',
        'cross_entropy', 'bal_accuracy' and 'avg_precision' are also supported. Score types for which smaller
        values are better, such as 'mae', 'rmse' and 'cross_entropy' are mapped to their negative counterparts.

        nreps (int): Number of repetitions of the permutation and rescoring procedure to perform for each feature; the
        importance values returned will be averages over repetitions. More repetitions will yield better importance
        estimates at the cost of greater computing time.

        nworkers (int): Number of parallel worker threads to use for permutation and rescoring.

        result_file (str): Optional path to a CSV file to which the importance table will be written.

    Returns:
        imp_df (DataFrame): Table of features and importance metrics. The table will include the columns returned by
        `base_feature_importance`, along with the permutation importance scores for each feature for the training, validation
        and test subsets.

    """
    log = logging.getLogger('ATOM')
    imp_df, model_pipeline, pparams = base_feature_importance(model_pipeline, params)

    # Compute the permutation-based importance values for the training, validation and test sets
    estimator = _get_estimator(model_pipeline)
    if score_type is None:
        score_type = pparams.model_choice_score_type
    scorer = _get_scorer(score_type)

    # Get the training, validation and test sets (we assume we're not using K-fold CV). These are DeepChem Dataset objects.
    (train_dset, valid_dset) = model_pipeline.data.train_valid_dsets[0]
    test_dset = model_pipeline.data.test_dset
    subsets = dict(train=train_dset, valid=valid_dset, test=test_dset)
    for subset, dset in subsets.items():
        log.debug(f"Computing permutation importance for {subset} set...")
        pi_result = permutation_importance(estimator, dset.X, dset.y, scoring=scorer, n_repeats=nreps, 
            random_state=17, n_jobs=nworkers)
        imp_df[f"{subset}_perm_importance_mean"] = pi_result['importances_mean']
        imp_df[f"{subset}_perm_importance_std"] = pi_result['importances_std']
    imp_df = imp_df.sort_values(by='valid_perm_importance_mean', ascending=False)
    if result_file is not None:
        imp_df.to_csv(result_file, index=False)
        log.info(f"Wrote importance table to {result_file}")
    return imp_df

# ===================================================================================================
def plot_feature_importances(imp_df, importance_col='valid_perm_importance_mean', max_feat=20, ascending=False):
    """Display a horizontal bar plot showing the relative importances of the most important features or feature clusters, according to
    the results of `permutation_feature_importance`, `cluster_permutation_importance` or a similar function.

    Args:
        imp_df (DataFrame): Table of results from `permutation_feature_importance`, `cluster_permutation_importance`,
        `base_feature_importance` or a similar function.

        importance_col (str): Name of the column in `imp_df` to plot values from.

        max_feat (int): The maximum number of features or feature clusters to plot values for.

        ascending (bool): Should the features be ordered by ascending values of `importance_col`? Defaults to False; can be set True
        for p-values or something else where small values mean greater importance.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(20,15))
    fi_df = imp_df.sort_values(by=importance_col,  ascending=ascending)
    if 'cluster_id' in fi_df.columns.values.tolist():
        feat_col = 'features'
    else:
        feat_col = 'feature'
    ax = sns.barplot(x=importance_col, y=feat_col, data=fi_df.head(max_feat))

# ===================================================================================================
def display_feature_clusters(model_pipeline=None, params=None, clust_height=1, 
                                   corr_file=None, show_matrix=False, show_dendro=True):
    """Cluster the input features used in the model specified by `model_pipeline` or `params`, using Spearman correlation
    as a similarity metric. Display a dendrogram and/or a correlation matrix heatmap, so the user can decide the
    height at which to cut the dendrogram in order to split the features into clusters, for input to
    `cluster_permutation_importance`.

    Args:
        model_pipeline (`ModelPipeline`): A pipeline object for a model that was trained in the current Python session
        or loaded from the model tracker or a tarball file. Either `model_pipeline` or `params` must be provided.

        params (`dict`): Parameter dictionary for a model to be trained and analyzed. Either `model_pipeline` or a
        `params` argument must be passed; if both are passed, `params` is ignored and the parameters from `model_pipeline`
        are used.

        clust_height (float): Height at which to draw a cut line in the dendrogram, to show how many clusters
        will be generated.

        corr_file (str): Path to an optional CSV file to be created containing the feature correlation matrix.

        show_matrix (bool): If True, plot a correlation matrix heatmap.

        show_dendro (bool): If True, plot the dendrogram.

    Returns:
        corr_linkage (np.ndarray): Linkage matrix from correlation clustering

    """
    log = logging.getLogger('ATOM')
    imp_df, model_pipeline, pparams = base_feature_importance(model_pipeline, params)
    features = imp_df.feature.values

    # Get the training, validation and test sets (we assume we're not using K-fold CV). These are DeepChem Dataset objects.
    (train_dset, valid_dset) = model_pipeline.data.train_valid_dsets[0]

    # Eliminate features that don't vary over the training set (and thus have zero importance)
    feat_idx = []
    for i, feat in enumerate(features):
        if len(set(train_dset.X[:,i])) > 1:
            feat_idx.append(i)
        else:
            log.debug(f"Removed unvarying feature {feat}")
    feat_idx = np.array(feat_idx, dtype=int)
    clust_X = train_dset.X[:,feat_idx]
    imp_df = imp_df.iloc[feat_idx]
    var_features = imp_df.feature.values

    # Cluster the training set features
    corr = spearmanr(clust_X, nan_policy='omit').correlation
    corr_df = pd.DataFrame(dict(feature=var_features))
    for i, feat in enumerate(var_features):
        corr_df[feat] = corr[:,i]
    if corr_file is not None:
        corr_df.to_csv(corr_file, index=False)
        log.info(f"Wrote correlation matrix to {corr_file}")
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, clust_height, criterion='distance')
    log.info(f"Cutting dendrogram at height {clust_height} yields {len(set(cluster_ids))} clusters")

    if not show_dendro:
        dendro = hierarchy.dendrogram(corr_linkage, labels=var_features.tolist(), no_plot=True, leaf_rotation=90)
    else:
        fig, ax = plt.subplots(figsize=(25,10))
        dendro = hierarchy.dendrogram(corr_linkage, labels=var_features.tolist(), ax=ax, leaf_rotation=90)
        fig.tight_layout()
        # Plot horizontal dashed line at clust_height
        line = ax.axhline(clust_height, c='b', linestyle='--')

        plt.show()

    if show_matrix:
        fig, ax = plt.subplots(figsize=(25,25))
        dendro_idx = np.arange(0, len(dendro['ivl']))
        leaves = dendro['leaves']
        ax.imshow(corr[leaves, :][:, leaves])
        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(dendro['ivl'], rotation='vertical')
        ax.set_yticklabels(dendro['ivl'])
        fig.tight_layout()
        plt.show()

    return corr_linkage


# ===================================================================================================
def cluster_permutation_importance(model_pipeline=None, params=None, score_type=None, clust_height=1, 
                                   result_file=None, nreps=10, nworkers=1):
    """Divide the input features used in a model into correlated clusters, then assess the importance of the features
    by iterating over clusters, permuting the values of all the features in the cluster, and measuring the effect
    on the model performance metric given by score_type for the training, validation and test subsets.

    Args:
        model_pipeline (`ModelPipeline`): A pipeline object for a model that was trained in the current Python session
        or loaded from the model tracker or a tarball file. Either `model_pipeline` or `params` must be provided.

        params (`dict`): Parameter dictionary for a model to be trained and analyzed. Either `model_pipeline` or a
        `params` argument must be passed; if both are passed, `params` is ignored and the parameters from `model_pipeline`
        are used.

        clust_height (float): Height at which to cut the dendrogram branches to split features into clusters.

        result_file (str): Path to a CSV file where a table of features and cluster indices will be written.

        nreps (int): Number of repetitions of the permutation and rescoring procedure to perform for each feature; the
        importance values returned will be averages over repetitions. More repetitions will yield better importance
        estimates at the cost of greater computing time.

        nworkers (int): Number of parallel worker threads to use for permutation and rescoring. Currently ignored; multithreading
        will be added in a future version.

    Returns:
        imp_df (DataFrame): Table of feature clusters and importance values

    """
    log = logging.getLogger('ATOM')
    imp_df, model_pipeline, pparams = base_feature_importance(model_pipeline, params)
    features = imp_df.feature.values

    # Compute the permutation-based importance values for the training, validation and test sets
    estimator = _get_estimator(model_pipeline)
    if score_type is None:
        score_type = pparams.model_choice_score_type
    scorer = _get_scorer(score_type)

    # Get the training, validation and test sets (we assume we're not using K-fold CV). These are DeepChem Dataset objects.
    (train_dset, valid_dset) = model_pipeline.data.train_valid_dsets[0]
    test_dset = model_pipeline.data.test_dset

    # Eliminate features that don't vary over the training set (and thus have zero importance)
    feat_idx = []
    for i, feat in enumerate(features):
        if len(set(train_dset.X[:,i])) > 1:
            feat_idx.append(i)
    feat_idx = np.array(feat_idx, dtype=int)
    clust_X = train_dset.X[:,feat_idx]
    imp_df = imp_df.iloc[feat_idx]
    var_features = imp_df.feature.values

    # Cluster the training set features
    corr = spearmanr(clust_X, nan_policy='omit').correlation
    corr_df = pd.DataFrame(dict(feature=var_features))
    for i, feat in enumerate(var_features):
        corr_df[feat] = corr[:,i]
    corr_linkage = hierarchy.ward(corr)

    cluster_ids = hierarchy.fcluster(corr_linkage, clust_height, criterion='distance')
    clust_to_feat_ids = defaultdict(list)
    clust_to_feat_names = defaultdict(list)
    for i, cluster_id in enumerate(cluster_ids):
        # clust_to_feat_ids will contain indices in original feature list
        clust_to_feat_ids[cluster_id].append(feat_idx[i])
        clust_to_feat_names[cluster_id].append(var_features[i])
    clust_idx = sorted(list(clust_to_feat_ids.keys()))
    clust_sizes = np.array([len(clust_to_feat_ids[clust]) for clust in clust_idx])
    clust_labels = [';'.join(clust_to_feat_names[clust]) for clust in clust_idx]
    n_non_sing = sum(clust_sizes > 1)
    log.info(f"Cutting dendrogram at height {clust_height} yields {len(set(cluster_ids))} clusters")
    log.info(f"{n_non_sing} are non-singletons")
    clust_df = pd.DataFrame(dict(cluster_id=clust_idx, num_feat=clust_sizes, features=clust_labels))
    clust_df = clust_df.sort_values(by='num_feat', ascending=False)

    # Now iterate through clusters; for each cluster, permute all the features in the cluster
    subsets = dict(train=train_dset, valid=valid_dset, test=test_dset)
    for subset, dset in subsets.items():
        log.debug(f"Computing permutation importance for {subset} set...")
        # First the score without permuting anything
        baseline_score = scorer(estimator, dset.X, dset.y)
        log.debug(f"Baseline {subset} {score_type} score = {baseline_score}")
        random_state = np.random.RandomState(17)
        #random_seed = random_state.randint(np.iinfo(np.int32).max + 1)
        importances_mean = []
        importances_std = []
        for clust in clust_df.cluster_id.values:
            scores = _calc_cluster_permutation_scores(estimator, dset.X, dset.y, clust_to_feat_ids[clust],
                                                      random_state, nreps, scorer)
            importances_mean.append(baseline_score - np.mean(scores))
            importances_std.append(np.std(scores))

        clust_df[f"{subset}_perm_importance_mean"] = importances_mean
        clust_df[f"{subset}_perm_importance_std"] = importances_std

    imp_df = clust_df.sort_values(by='valid_perm_importance_mean', ascending=False)
    if result_file is not None:
        imp_df.to_csv(result_file, index=False)
        log.info(f"Wrote cluster importances to {result_file}")
    return imp_df

# ===================================================================================================
def _calc_cluster_permutation_scores(estimator, X, y, col_indices, random_state, n_repeats, scorer):
    """Calculate score of estimator when `col_indices` are all permuted randomly."""
    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. 
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        for col_idx in col_indices:
            random_state.shuffle(shuffling_idx)
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        feature_score = scorer(estimator, X_permuted, y)
        scores[n_round] = feature_score

    return scores



