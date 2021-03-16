"""
Plotting routines for visualizing performance of regression and classification models
"""

import os

import matplotlib

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import umap
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D

from atomsci.ddm.pipeline import perf_data as perf

#matplotlib.style.use('ggplot')
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize=12)


#------------------------------------------------------------------------------------------------------------------------
def plot_pred_vs_actual(MP, epoch_label='best', threshold=None, error_bars=False, pdf_dir=None):
    """
    Plot predicted vs actual values from regression model for the specified dataset subset (train, 
    valid, test or full) for the dataset in ModelPipeline MP.  If threshold is given, draw dashed lines
    in both directions (e.g. to indicate an activity threshold). If pdf_dir is given, output the
    plots to PDF files in the given directory; otherwise they are drawn on the current output device. If 
    uncertainty estimates are included in the model predictions and error_bars is True, error bars will be 
    drawn at +- 1 SD from predicted y values.
    """
    params = MP.params
    # For now restrict this to regression models. 
    # TODO: Implement a version of plot_pred_vs_actual for classification models.
    if params.prediction_type != 'regression':
        MP.log.error("plot_pred_vs_actual is currently for regression models only.")
        return
    wrapper = MP.model_wrapper
    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_%s_%s_%s_pred_vs_actual.pdf' % (params.dataset_name, params.model_type,
                                params.featurizer, params.splitter))
        pdf = PdfPages(pdf_path)

    if MP.run_mode == 'training':
        subsets = ['train', 'valid', 'test']
    else:
        subsets = ['full']
    dataset_name = MP.data.dataset_name
    splitter = MP.params.splitter
    model_type = MP.params.model_type
    featurizer = MP.params.featurizer
    tasks = MP.params.response_cols
    for subset in subsets:
        perf_data = wrapper.get_perf_data(subset, epoch_label)
        pred_results = perf_data.get_prediction_results()
        y_actual = perf_data.get_real_values()
        ids, y_pred, y_std = perf_data.get_pred_values()
        r2 = pred_results['r2_score']
        if perf_data.num_tasks > 1:
            r2_scores = pred_results['task_r2_scores']
        else:
            r2_scores = [r2]
        for t in range(MP.params.num_model_tasks):
            fig, ax = plt.subplots(figsize=(12.0,12.0))
            title = '%s\n%s split %s model on %s features\n%s subset predicted vs actual %s, R^2 = %.3f' % (
                     dataset_name, splitter, model_type, featurizer, subset, tasks[t], r2_scores[t])
            # Force axes to have same scale
            ymin = min(min(y_actual[:,t]), min(y_pred[:,t]))
            ymax = max(max(y_actual[:,t]), max(y_pred[:,t]))
            ax.set_xlim(ymin, ymax)
            ax.set_ylim(ymin, ymax)
            if error_bars and y_std is not None:
                # Draw error bars
                ax.errorbar(y_actual[:,t], y_pred[:,t], y_std[:,t], c='blue', marker='o', alpha=0.4, linestyle='')
            else:
                plt.scatter(y_actual[:,t], y_pred[:,t], s=9, c='blue', marker='o', alpha=0.4)
            ax.set_xlabel('Observed value')
            ax.set_ylabel('Predicted value')
            # Draw an identity line
            ax.plot([ymin,ymax], [ymin,ymax], c='forestgreen', linestyle='--')
            if threshold is not None:
                plt.axvline(threshold, color='r', linestyle='--')
                plt.axhline(threshold, color='r', linestyle='--')
            ax.set_title(title, fontdict={'fontsize' : 10})
            if pdf_dir is not None:
                pdf.savefig(fig)
    if pdf_dir is not None:
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)


#------------------------------------------------------------------------------------------------------------------------
def plot_perf_vs_epoch(MP, pdf_dir=None):
    """
    Plot the current NN model's standard performance metric (r2_score or roc_auc_score) vs epoch number for the training and 
    validation subsets. If the model was trained with k-fold CV, plot error bars or shading out to += 1 SD from the mean
    score metric values. Also plot the validation set score used for ranking training epochs and other hyperparameters 
    against epoch number.
    """
    wrapper = MP.model_wrapper
    if 'train_epoch_perfs' not in wrapper.__dict__:
        raise ValueError("plot_perf_vs_epoch() can only be called for NN models")
    num_epochs = wrapper.num_epochs_trained
    subset_perf = dict(training = wrapper.train_epoch_perfs[:num_epochs], validation = wrapper.valid_epoch_perfs[:num_epochs], 
                       test = wrapper.test_epoch_perfs[:num_epochs])
    subset_std = dict(training = wrapper.train_epoch_perf_stds[:num_epochs], validation = wrapper.valid_epoch_perf_stds[:num_epochs],
                       test = wrapper.test_epoch_perf_stds[:num_epochs])
    num_folds = len(MP.data.train_valid_dsets)
    model_scores = wrapper.model_choice_scores[:num_epochs]
    model_score_type = MP.params.model_choice_score_type
    best_epoch = wrapper.best_epoch

    epoch = list(range(num_epochs))
    if MP.params.prediction_type == 'regression':
        perf_label = 'R-squared'
    else:
        perf_label = 'ROC AUC'

    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_perf_vs_epoch.pdf' % os.path.basename(MP.params.output_dir))
        pdf = PdfPages(pdf_path)
    subset_colors = dict(training='blue', validation='forestgreen', test='red')
    subset_shades = dict(training='deepskyblue', validation='lightgreen', test='hotpink')
    fig, ax = plt.subplots(figsize=(10,10))
    title = '%s dataset\n%s vs epoch for %s %s model on %s features with %s split\nBest validation set performance at epoch %d' % (
            MP.params.dataset_name, perf_label, MP.params.model_type,  MP.params.prediction_type,
            MP.params.featurizer,  MP.params.splitter,  best_epoch)
    for subset in ['training', 'validation', 'test']:
        ax.plot(epoch, subset_perf[subset], color=subset_colors[subset], label=subset)
        # Add shading to show variance across folds
        if num_folds > 1:
            ax.fill_between(epoch, subset_perf[subset] + subset_std[subset], subset_perf[subset] - subset_std[subset],
                            alpha=0.3, facecolor=subset_shades[subset], linewidth=0)
    plt.axvline(best_epoch, color='forestgreen', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(perf_label)
    ax.set_title(title, fontdict={'fontsize' : 12})
    legend = ax.legend(loc='lower right')
    if pdf_dir is not None:
        pdf.savefig(fig)

    # Now plot the score used for choosing the best epoch and model params
    fig, ax = plt.subplots(figsize=(10,10))
    title = '%s dataset\n%s vs epoch for %s %s model on %s features with %s split\nBest validation set performance at epoch %d' % (
            MP.params.dataset_name, model_score_type, MP.params.model_type,  MP.params.prediction_type,
            MP.params.featurizer,  MP.params.splitter,  best_epoch)
    ax.plot(epoch, model_scores, color=subset_colors['validation'])
    plt.axvline(best_epoch, color='red', linestyle='--')
    ax.set_xlabel('Epoch')
    if model_score_type in perf.loss_funcs:
        score_label = "negative %s" % model_score_type
    else:
        score_label = model_score_type
    ax.set_ylabel(score_label)
    ax.set_title(title, fontdict={'fontsize' : 12})
    if pdf_dir is not None:
        pdf.savefig(fig)
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)



#------------------------------------------------------------------------------------------------------------------------
def get_perf_curve_data(MP, epoch_label, curve_type='ROC'):
    """
    Common code for ROC and precision-recall curves. Returns true classes and active class probabilities
    for each training/test data subset.
    """
    if MP.params.prediction_type != 'classification':
        MP.log.error("Can only plot %s curve for classification models" % curve_type)
        return {}

    if MP.run_mode == 'training':
        subsets = ['train', 'valid', 'test']
    else:
        subsets = ['full']
    wrapper = MP.model_wrapper
    curve_data = {}
    for subset in subsets:
        perf_data = wrapper.get_perf_data(subset, epoch_label)
        true_classes = perf_data.get_real_values()
        ids, pred_classes, class_probs, prob_stds = perf_data.get_pred_values()
        ntasks = class_probs.shape[1]
        nclasses = class_probs.shape[-1]
        if nclasses != 2:
            MP.log.error("%s curve plot is only supported for binary classifiers" % curve_type)
            return {}
        prob_active = class_probs[:,:,1]
        roc_aucs = [metrics.roc_auc_score(true_classes[:,i], prob_active[:,i], average='macro') 
                    for i in range(ntasks)]
        prc_aucs = [metrics.average_precision_score(true_classes[:,i], prob_active[:,i], average='macro') 
                    for i in range(ntasks)]
        curve_data[subset] = dict(true_classes=true_classes, prob_active=prob_active, roc_aucs=roc_aucs, prc_aucs=prc_aucs)
    return curve_data

#------------------------------------------------------------------------------------------------------------------------
def plot_ROC_curve(MP, epoch_label='best', pdf_dir=None):
    """
    Plot ROC curves for a classification model.
    """
    params = MP.params
    curve_data = get_perf_curve_data(MP, epoch_label, 'ROC')
    if len(curve_data) == 0:
        return
    if MP.run_mode == 'training':
        # Draw overlapping ROC curves for train, valid and test sets
        subsets = ['train', 'valid', 'test']
    else:
        subsets = ['full']
    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_%s_model_%s_features_%s_split_ROC_curves.pdf' % (
                                params.dataset_name, params.model_type, params.featurizer, params.splitter))
        pdf = PdfPages(pdf_path)
    subset_colors = dict(train='blue', valid='forestgreen', test='red', full='purple')
    # For multitask, do a separate figure for each task
    ntasks = curve_data[subsets[0]]['prob_active'].shape[1]
    for i in range(ntasks):
        fig, ax = plt.subplots(figsize=(10,10))
        title = '%s dataset\nROC curve for %s %s classifier on %s features with %s split' % (
                           params.dataset_name, params.response_cols[i], 
                           params.model_type, params.featurizer, params.splitter)
        for subset in subsets:
            fpr, tpr, thresholds = metrics.roc_curve(curve_data[subset]['true_classes'][:,i],
                                                     curve_data[subset]['prob_active'][:,i])
      
            roc_auc = curve_data[subset]['roc_aucs'][i]
            ax.step(fpr, tpr, color=subset_colors[subset], label="%s: AUC = %.3f" % (subset, roc_auc))
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(title, fontdict={'fontsize' : 12})
        legend = ax.legend(loc='lower right')
    
        if pdf_dir is not None:
            pdf.savefig(fig)
    if pdf_dir is not None:
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)


#------------------------------------------------------------------------------------------------------------------------
def plot_prec_recall_curve(MP, epoch_label='best', pdf_dir=None):
    """
    Plot precision-recall curves for a classification model.
    """
    params = MP.params
    curve_data = get_perf_curve_data(MP, epoch_label, 'precision-recall')
    if len(curve_data) == 0:
        return
    if MP.run_mode == 'training':
        # Draw overlapping PR curves for train, valid and test sets
        subsets = ['train', 'valid', 'test']
    else:
        subsets = ['full']
    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_%s_model_%s_features_%s_split_PRC_curves.pdf' % (
                                params.dataset_name, params.model_type, params.featurizer, params.splitter))
        pdf = PdfPages(pdf_path)
    subset_colors = dict(train='blue', valid='forestgreen', test='red', full='purple')
    # For multitask, do a separate figure for each task
    ntasks = curve_data[subsets[0]]['prob_active'].shape[1]
    for i in range(ntasks):
        fig, ax = plt.subplots(figsize=(10,10))
        title = '%s dataset\nPrecision-recall curve for %s %s classifier on %s features with %s split' % (
                           params.dataset_name, params.response_cols[i], 
                           params.model_type, params.featurizer, params.splitter)
        for subset in subsets:
            precision, recall, thresholds = metrics.precision_recall_curve(curve_data[subset]['true_classes'][:,i],
                                                     curve_data[subset]['prob_active'][:,i])
      
            prc_auc = curve_data[subset]['prc_aucs'][i]
            ax.step(recall, precision, color=subset_colors[subset], label="%s: AUC = %.3f" % (subset, prc_auc))
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title, fontdict={'fontsize' : 12})
        legend = ax.legend(loc='lower right')
    
        if pdf_dir is not None:
            pdf.savefig(fig)
    if pdf_dir is not None:
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)

#------------------------------------------------------------------------------------------------------------------------
def plot_umap_feature_projections(MP, ndim=2, num_neighbors=20, min_dist=0.1, 
                                  fit_to_train=True,
                                  dist_metric='euclidean', dist_metric_kwds={}, 
                                  target_weight=0, random_seed=17, pdf_dir=None):
    """
    Take numeric features (descriptors, fingerprints, etc.) input to a model, project them to
    2 or 3 dimensions, and draw a scatterplot of the projected coordinates, with markers shape-coded
    to indicate whether the associated compound was in the training, validation or test set. For classification
    models, also use the marker shape to indicate whether the compound's class was correctly predicted, and use
    color to indicate whether the true class was active or inactive. For regression models, use the marker color
    to indicate the discrepancy between the predicted and actual values.
    """
    if (ndim != 2) and (ndim != 3):
      MP.log.error('Only 2D and 3D visualizations are supported by plot_umap_feature_projections()')
      return
    params = MP.params
    if params.featurizer == 'graphconv':
        MP.log.error('plot_umap_feature_projections() does not support GraphConv models.')
        return
    split_strategy = params.split_strategy
    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_%s_model_%s_split_umap_%s_%dd_projection.pdf' % (
                                params.dataset_name, params.model_type, params.splitter, params.featurizer, ndim))
        pdf = PdfPages(pdf_path)

    dataset = MP.data.dataset
    ncmpds = dataset.y.shape[0]
    ntasks = dataset.y.shape[1]
    nfeat = dataset.X.shape[1]
    cmpd_ids = {}
    features = {}
    # TODO: Need an option to pass in a training (or combined training & validation) dataset, in addition
    # to a dataset on which we ran predictions with the same model, to display as training and test data.

    if split_strategy == 'train_valid_test':
        subsets = ['train', 'valid', 'test']
        train_dset, valid_dset = MP.data.train_valid_dsets[0]
        features['train'] = train_dset.X
        cmpd_ids['train'] = train_dset.ids
    else:
        # For k-fold split, every compound in combined training & validation set is in the validation
        # set for 1 fold and in the training set for the k-1 others.
        subsets = ['valid', 'test']
        features['train'] = np.empty((0,nfeat))
        cmpd_ids['train'] = []
        valid_dset = MP.data.combined_train_valid_data
    features['valid'] = valid_dset.X
    cmpd_ids['valid'] = valid_dset.ids

    test_dset = MP.data.test_dset
    features['test'] = test_dset.X
    cmpd_ids['test'] = test_dset.ids

    all_features = np.concatenate([features[subset] for subset in subsets], axis=0)
    if fit_to_train:
        if split_strategy == 'train_valid_test':
            fit_features = features['train']
        else:
            fit_features = features['valid']
    else:
        fit_features = all_features

    epoch_label = 'best'
    pred_vals = {}
    real_vals = {}
    for subset in subsets:
        perf_data = MP.model_wrapper.get_perf_data(subset, epoch_label)
        y_actual = perf_data.get_real_values()
        if MP.params.prediction_type == 'classification':
            ids, y_pred, class_probs, y_std = perf_data.get_pred_values()
        else:
            ids, y_pred, y_std = perf_data.get_pred_values()
        # Have to get predictions and real values in same order as in dataset subset
        pred_dict = dict([(id, y_pred[i,:]) for i, id in enumerate(ids)])
        real_dict = dict([(id, y_actual[i,:]) for i, id in enumerate(ids)])
        pred_vals[subset] = np.concatenate([pred_dict[id] for id in cmpd_ids[subset]], axis=0)
        real_vals[subset] = np.concatenate([real_dict[id] for id in cmpd_ids[subset]], axis=0)

    all_actual = np.concatenate([real_vals[subset] for subset in subsets], axis=0).reshape((-1,ntasks))
    if fit_to_train:
        if split_strategy == 'train_valid_test':
            fit_actual = real_vals['train'].reshape((-1,ntasks))
        else:
            fit_actual = real_vals['valid'].reshape((-1,ntasks))
    else:
        fit_actual = all_actual
    preds = np.concatenate([pred_vals[subset] for subset in subsets], axis=0).reshape((-1,ntasks))
    compound_ids = np.concatenate([cmpd_ids[subset] for subset in subsets], axis=None)


    if MP.params.prediction_type == 'classification':
        target_metric = 'categorical'
    else:
        target_metric = 'euclidean'

    for i in range(ntasks):
        mapper = umap.UMAP(n_neighbors=num_neighbors, n_components=ndim, metric=dist_metric, target_metric=target_metric,
                           metric_kwds=dist_metric_kwds, target_weight=target_weight, random_state=random_seed)
        # Ideally, the mapper should be fit to the training data only, and then applied to the validation and test sets
        mapper.fit(fit_features, y=fit_actual[:,i])
        proj = mapper.transform(all_features)
        proj_cols = ['umap_X', 'umap_Y']
        if ndim == 3:
            proj_cols.append('umap_Z')
        proj_df = pd.DataFrame.from_records(proj, columns=proj_cols)
        proj_df['compound_id'] = compound_ids
        proj_df['actual'] = all_actual[:,i]
        dset_subset = ['training']*len(cmpd_ids['train']) + ['valid']*len(cmpd_ids['valid']) + ['test']*len(cmpd_ids['test'])
    
        if dist_metric == 'minkowski':
          metric_name = 'minkowski(%.2f)' % dist_metric_kwds['p']
        else:
          metric_name = dist_metric

        fig = plt.figure(figsize=(15,15))
        title = '%s dataset\n%s features projected to %dD with %s metric\n%d neighbors, min_dist = %.3f' % (
                      params.dataset_name, params.featurizer, ndim, metric_name, num_neighbors, min_dist)
    
        if MP.params.prediction_type == 'classification':
            is_correct = (all_actual[:,i] == preds[:,i]).astype(np.int32)
            result = [('incorrect', 'correct')[i] for i in is_correct]
            # Mark predictions as correct or incorrect; check that class representations are the same.
            proj_df['subset'] = ['%s/%s' % vals for vals in zip(dset_subset,result)]
            green_red_pal = {0 : 'forestgreen', 1 : 'red'}
            marker_map = {'training/correct' : 'o', 'training/incorrect' : 's', 
                        'valid/correct' : '^', 'valid/incorrect' : 'v', 
                        'test/correct' : 'P', 'test/incorrect' : '*'}
            size_map = {'training/correct' : 49, 'training/incorrect' : 64, 
                        'valid/correct' : 81, 'valid/incorrect' : 125, 
                        'test/correct' : 125, 'test/incorrect' : 225}
            if ndim == 2:
                ax = fig.add_subplot(111)
                sns.scatterplot(x='umap_X', y='umap_Y', hue='actual', palette=green_red_pal, 
                        style='subset', markers=marker_map,
                        style_order=['training/correct', 'training/incorrect', 'valid/correct', 'valid/incorrect',
                                     'test/correct', 'test/incorrect' ],
                        size='subset', sizes=size_map,
                        data=proj_df, ax=ax)
            elif ndim == 3:
                ax = fig.add_subplot(111, projection='3d')
                colors = [green_red_pal[a] for a in proj_df.actual.values]
                markers = [marker_map[subset] for subset in proj_df.subset.values]
                #ax.scatter(proj_df.umap_X.values, proj_df.umap_Y.values, proj_df.umap_Z.values, c=colors, m=markers, s=49)
                ax.scatter(proj_df.umap_X.values, proj_df.umap_Y.values, proj_df.umap_Z.values, c=colors, s=49)
        else:
            # regression model
            proj_df['subset'] = dset_subset
            proj_df['error'] = preds[:,i] - all_actual[:,i]
            marker_map = {'training' : 'o', 'valid' : 'v', 'test' : 'P'}
            ncol = 12
            blue_red_pal = sns.blend_palette(['red', 'green', 'blue'], 12, as_cmap=True)
            if ndim == 2:
                ax = fig.add_subplot(111)
                #sns.scatterplot(x='umap_X', y='umap_Y', hue='error',
                sns.scatterplot(x='umap_X', y='umap_Y', hue='error', palette=blue_red_pal, 
                        size='actual', sizes=(49,144), alpha=0.95,
                        style='subset', markers=marker_map, style_order=['training', 'valid', 'test'],
                        data=proj_df, ax=ax)
            elif ndim == 3:
                ax = fig.add_subplot(111, projection='3d')
                # Map errors to palette indices
                errs = proj_df.error.values.astype(np.int32)
                ind = 1 + errs - min(errs)
                ind[ind >= ncol] = ncol-1
                colors = [blue_red_pal[i] for i in ind]
                markers = [marker_map[subset] for subset in proj_df.subset.values]
                ax.scatter(proj_df.umap_X.values, proj_df.umap_Y.values, proj_df.umap_Z.values, c=colors, m=markers, s=49)
    
    
        ax.set_title(title, fontdict={'fontsize' : 10})
    if pdf_dir is not None:
        pdf.savefig(fig)
    if pdf_dir is not None:
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)

#------------------------------------------------------------------------------------------------------------------------
def plot_umap_train_set_neighbors(MP, num_neighbors=20, min_dist=0.1, 
                                  dist_metric='euclidean', dist_metric_kwds={}, 
                                  random_seed=17, pdf_dir=None):
    """
    Project features of whole dataset to 2 dimensions, without regard to response values. Plot training & validation set
    or training and test set compounds, color- and symbol-coded according to actual classification and split set.
    The plot does not take predicted values into account at all. Does not work with regression data.
    """
    ndim = 2
    params = MP.params
    if params.prediction_type != 'classification':
        MP.log.error("plot_umap_train_set_neighbors() doesn't work with regression data")
        return
    if params.featurizer == 'graphconv':
        MP.log.error('plot_umap_train_set_neighbors() does not support GraphConv models.')
        return
    split_strategy = params.split_strategy
    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_%s_model_%s_split_umap_%s_%dd_projection.pdf' % (
                                params.dataset_name, params.model_type, params.splitter, params.featurizer, ndim))
        pdf = PdfPages(pdf_path)

    dataset = MP.data.dataset
    ncmpds = dataset.y.shape[0]
    ntasks = dataset.y.shape[1]
    nfeat = dataset.X.shape[1]
    cmpd_ids = {}
    features = {}
    # TODO: Need an option to pass in a training (or combined training & validation) dataset, in addition
    # to a dataset on which we ran predictions with the same model, to display as training and test data.

    if split_strategy == 'train_valid_test':
        subsets = ['train', 'valid', 'test']
        train_dset, valid_dset = MP.data.train_valid_dsets[0]
        features['train'] = train_dset.X
        cmpd_ids['train'] = train_dset.ids
    else:
        # For k-fold split, every compound in combined training & validation set is in the validation
        # set for 1 fold and in the training set for the k-1 others.
        subsets = ['valid', 'test']
        features['train'] = np.empty((0,nfeat))
        cmpd_ids['train'] = []
        valid_dset = MP.data.combined_train_valid_data
    features['valid'] = valid_dset.X
    cmpd_ids['valid'] = valid_dset.ids

    test_dset = MP.data.test_dset
    features['test'] = test_dset.X
    cmpd_ids['test'] = test_dset.ids

    all_features = np.concatenate([features[subset] for subset in subsets], axis=0)

    epoch_label = 'best'
    pred_vals = {}
    real_vals = {}
    for subset in subsets:
        perf_data = MP.model_wrapper.get_perf_data(subset, epoch_label)
        y_actual = perf_data.get_real_values()
        ids, y_pred, class_probs, y_std = perf_data.get_pred_values()
        # Have to get predictions and real values in same order as in dataset subset
        real_dict = dict([(id, y_actual[i,:]) for i, id in enumerate(ids)])
        real_vals[subset] = np.concatenate([real_dict[id] for id in cmpd_ids[subset]], axis=0)

    all_actual = np.concatenate([real_vals[subset] for subset in subsets], axis=0).reshape((-1,ntasks))
    compound_ids = np.concatenate([cmpd_ids[subset] for subset in subsets], axis=None)


    for i in range(ntasks):
        mapper = umap.UMAP(n_neighbors=num_neighbors, n_components=ndim, metric=dist_metric,
                           metric_kwds=dist_metric_kwds, random_state=random_seed)
        mapper.fit(all_features)
        proj = mapper.transform(all_features)
        proj_cols = ['umap_X', 'umap_Y']
        proj_df = pd.DataFrame.from_records(proj, columns=proj_cols)
        proj_df['compound_id'] = compound_ids
        proj_df['actual'] = all_actual[:,i]
        dset_subset = ['train']*len(cmpd_ids['train']) + ['valid']*len(cmpd_ids['valid']) + ['test']*len(cmpd_ids['test'])
        proj_df['dset_subset'] = dset_subset
    
        if dist_metric == 'minkowski':
          metric_name = 'minkowski(%.2f)' % dist_metric_kwds['p']
        else:
          metric_name = dist_metric

        if params.featurizer == 'ecfp':
            feat_type = 'ECFP'
        elif params.featurizer == 'descriptors':
            feat_type = 'precomputed %s descriptor' % params.descriptor_type
        elif params.featurizer == 'computed_descriptors':
            feat_type = 'computed %s descriptor' % params.descriptor_type
        else:
            feat_type = params.featurizer
            
    
        classif = np.array(['inactive']*proj_df.shape[0])
        classif[proj_df.actual == 1] = 'active'
        proj_df['classif'] = classif
        proj_df['subset'] = ['%s/%s' % vals for vals in zip(dset_subset,classif)]
        for subset in ['valid', 'test']:
            fig, ax = plt.subplots(figsize=(15,15))
            proj_plt_df = proj_df[(proj_df.dset_subset == 'train') | (proj_df.dset_subset == subset)]
            if subset == 'valid':
                pal = {'train/active' : 'forestgreen', 'train/inactive' : 'red', 
                            'valid/active' : 'blue', 'valid/inactive' : 'magenta'}
                marker_map = {'train/active' : 'o', 'train/inactive' : 's', 
                            'valid/active' : '*', 'valid/inactive' : 'v'}
                size_map = {'train/active' : 64, 'train/inactive' : 49, 
                            'valid/active' : 192, 'valid/inactive' : 81}
                style_order=['train/inactive', 'valid/inactive', 'train/active', 'valid/active' ]
            else:
                pal = {'train/active' : 'forestgreen', 'train/inactive' : 'red', 
                            'test/active' : 'blue', 'test/inactive' : 'magenta'}
                marker_map = {'train/active' : 'o', 'train/inactive' : 's', 
                            'test/active' : '*', 'test/inactive' : 'v'}
                size_map = {'train/active' : 64, 'train/inactive' : 49, 
                            'test/active' : 192, 'test/inactive' : 81}
                style_order=['train/inactive', 'test/inactive', 'train/active', 'test/active' ]
            sns.scatterplot(x='umap_X', y='umap_Y', hue='subset', palette=pal,
                        style='subset', markers=marker_map,
                        style_order=style_order, size='subset', sizes=size_map,
                        data=proj_plt_df, ax=ax)
            title = '%s dataset\n%s features projected to 2D with %s metric\nTraining and %s subsets from %s splitter' % (
                      params.dataset_name, feat_type, metric_name, subset, params.splitter)
            ax.set_title(title, fontdict={'fontsize' : 10})
            plt.show()
            if pdf_dir is not None:
                pdf.savefig(fig)
    if pdf_dir is not None:
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)

