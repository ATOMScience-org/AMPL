"""Plotting routines for visualizing performance of regression and classification models"""

import os

import matplotlib

import tempfile
import tarfile
import json
import pandas as pd
import numpy as np
import seaborn as sns
import umap
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from atomsci.ddm.utils import file_utils as futils
from atomsci.ddm.utils import model_file_reader as mfr
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import perf_data as perf
from atomsci.ddm.pipeline import predict_from_model as pfm

#matplotlib.style.use('ggplot')
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize=12)

#------------------------------------------------------------------------------------------------------------------------
# Labels used in plot titles and axis labels
score_type_label = dict(r2 = '$R^2$', mae = 'MAE', rmse = 'RMSE',
        roc_auc = 'ROC AUC', precision = 'Precision', ppv = 'Precision', recall = 'Recall',
        npv = 'NPV', cross_entropy = 'Cross entropy', accuracy = 'Accuracy', bal_accuracy = 'Balanced accuracy',
        avg_precision = 'Average precision', prc_auc = 'Average precision', 
        MCC = 'Matthews corr coef', mcc = 'Matthews corr coef', kappa = "Cohen's kappa")

#------------------------------------------------------------------------------------------------------------------------
# CVD-friendly plot colors. Use the more saturated 'colorblind' palette for point and line plots and 'pastel' for shading.

# Map train/valid/test categories to colors and shades
sat_cols = sns.color_palette('colorblind').as_hex()
train_col = sat_cols[0]
valid_col = sat_cols[2]
test_col = sat_cols[1]
full_col = sat_cols[4]

shade_cols = sns.color_palette('pastel').as_hex()
train_shade = shade_cols[0]
valid_shade = shade_cols[2]
test_shade = shade_cols[1]

# Map boolean distinctions (false/true, incorrect/correct, etc.) to colors from 'colorblind' palette
binary_pal = {0 : sat_cols[3], 1 : sat_cols[0]}

# Map combined split subset / activity categories to colors from 'colorblind' palette
train_active_col = sat_cols[0]
train_inactive_col = sat_cols[9]
test_active_col = sat_cols[3]
test_inactive_col = sat_cols[2]

# Use sequential 'crest' palette, which runs from light green to dark blue, for continuous values
continuous_pal = sns.color_palette('crest', as_cmap=True)

#------------------------------------------------------------------------------------------------------------------------
def plot_pred_vs_actual(model, epoch_label='best', threshold=None, error_bars=False, plot_size=7, 
                       external_training_data=None, pdf_dir=None):
    """Plot predicted vs actual values from a trained regression model for each split subset (train,
    valid, and test).

    Args:
        model (`ModelPipeline` or str): Either a ModelPipeline object for a model that was trained in the current Python session,
        or a path to a saved model .tar.gz file or directory.
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
        threshold (float): Threshold activity value to mark on plot with dashed lines.
        error_bars (bool): If true and if uncertainty estimates are included in the model predictions, draw error bars
            at +- 1 SD from the predicted y values.
        plot_size (float): Height of each subplot
        external_training_data (str): Path to copy of training dataset, if different from path used when model was trained. Only
        used for saved models.
        pdf_dir (str): If given, output the plots to a PDF file in the given directory.

    Returns:
        None

    """
    if isinstance(model, str):
        return plot_pred_vs_actual_from_file(model, plot_size=plot_size, external_training_data=external_training_data)
    elif not isinstance(model, mp.ModelPipeline):
        raise ValueError('model must be either a ModelPipeline or a path to a saved model')
    params = model.params
    # For now restrict this to regression models. 
    if params.prediction_type != 'regression':
        model.log.error("plot_pred_vs_actual() should only be called for regression models. Please try plot_confusion_matrices() instead..")
        return
    wrapper = model.model_wrapper
    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_%s_%s_%s_pred_vs_actual.pdf' % (params.dataset_name, params.model_type,
                                params.featurizer, params.splitter))
        pdf = PdfPages(pdf_path)

    if model.run_mode == 'training':
        subsets = ['train', 'valid', 'test']
    else:
        subsets = ['full']
    num_ss = len(subsets)
    dataset_name = model.data.dataset_name
    splitter = model.params.splitter
    model_type = model.params.model_type
    featurizer = model.params.featurizer
    if featurizer in ('computed_descriptors', 'descriptors'):
        featurizer = f"{model.params.descriptor_type} descriptors"
    else:
        featurizer = f"{featurizer} features"
    tasks = model.params.response_cols
    num_tasks = len(tasks)
    if model.params.model_type != "hybrid":
        fig, axes = plt.subplots(num_tasks, num_ss, figsize=(plot_size*num_ss, plot_size*num_tasks))
        if num_ss > 1:
            suptitle = f"{dataset_name}  {splitter} {params.split_strategy} split {model_type} model on {featurizer} features, predicted vs actual values by split subset"
        else:
            suptitle = f"{model_type} model on {featurizer} features, predicted vs actual values on dataset {dataset_name}"
        fig.suptitle(suptitle, y=0.95)
        axes = axes.flatten()
        for t, resp in enumerate(tasks):
            for s, subset in enumerate(subsets):
                perf_data = wrapper.get_perf_data(subset, epoch_label)
                pred_results = perf_data.get_prediction_results()
                y_actual = perf_data.get_real_values()
                y_weights = perf_data.get_weights()
                y_actual=np.where(y_weights==0, np.nan, y_actual)
                ids, y_pred, y_std = perf_data.get_pred_values()
                r2 = pred_results['r2_score']
                if perf_data.num_tasks > 1:
                    r2_scores = pred_results['task_r2_scores']
                else:
                    r2_scores = [r2]
                ax_ind = num_ss*t + s
                ax = axes[ax_ind]
                ymin = min(np.nanmin(y_actual[:,t]), np.nanmin(y_pred[:,t]))
                ymax = max(np.nanmax(y_actual[:,t]), np.nanmax(y_pred[:,t]))
                # Force axes to have same scale for all subsets
                ax.set_xlim(ymin, ymax)
                ax.set_ylim(ymin, ymax)
                if error_bars and y_std is not None:
                    # Draw error bars
                    ax.errorbar(y_actual[:,t], y_pred[:,t], y_std[:,t], c=train_col, marker='o', alpha=0.4, linestyle='')
                else:
                    ax.scatter(y_actual[:,t], y_pred[:,t], s=25, marker='o', alpha=0.4)
                ax.set_xlabel(f"Actual {resp}")
                ax.set_ylabel(f"Predicted {resp}")
                # Draw an identity line
                ax.plot([ymin,ymax], [ymin,ymax], c=test_col, linestyle='--', alpha=0.75, zorder=0)
                # Draw threshold lines if requested
                if threshold is not None:
                    plt.axvline(threshold, color=test_col, linestyle='--')
                    plt.axhline(threshold, color=test_col, linestyle='--')
                # Set subplot title
                if s == 0:
                    subtitle = f"{resp} {subset}, {score_type_label['r2']} = {r2_scores[t]:.3f}"
                else:
                    subtitle = f"{subset} {score_type_label['r2']} = {r2_scores[t]:.3f}"
                ax.set_title(subtitle, fontdict={'fontsize' : 10})
                if pdf_dir is not None:
                    pdf.savefig(fig)
    else:
        # As yet, hybrid models are singletask only. There are two plots per row, one for Ki/IC50 data and the other for
        # % binding / inhibition data, with one row per subset.
        for s, subset in enumerate(subsets):
            perf_data = wrapper.get_perf_data(subset, epoch_label)
            pred_results = perf_data.get_prediction_results()
            y_actual = perf_data.get_real_values()
            ids, y_pred, y_std = perf_data.get_pred_values()
            r2 = pred_results['r2_score']
            if perf_data.num_tasks > 1:
                r2_scores = pred_results['task_r2_scores']
            else:
                r2_scores = [r2]
            fig, ax = plt.subplots(1,2, figsize=(20.0,10.0))
            title = '%s\n%s split %s model on %s features\n%s subset predicted vs actual %s, R^2 = %.3f' % (
                    dataset_name, splitter, model_type, featurizer, subset, "Ki/XC50", r2_scores[0])
            pos_ki = np.where(np.isnan(y_actual[:, 1]))[0]
            pos_bind = np.where(~np.isnan(y_actual[:, 1]))[0]
            y_pred_ki = y_pred[pos_ki, 0]
            y_real_ki = y_actual[pos_ki, 0]
            y_pred_bind = y_pred[pos_bind, 0]
            y_real_bind = y_actual[pos_bind, 0]
            ki_ymin = min(min(y_real_ki), min(y_pred_ki)) - 0.5
            ki_ymax = max(max(y_real_ki), max(y_pred_ki)) + 0.5
            ax[0].set_xlim(ki_ymin, ki_ymax)
            ax[0].set_ylim(ki_ymin, ki_ymax)
            ax[0].scatter(y_real_ki, y_pred_ki, s=9, c=train_col, marker='o', alpha=0.4)
            ax[0].set_xlabel('Observed value')
            ax[0].set_ylabel('Predicted value')
            ax[0].plot([ki_ymin,ki_ymax], [ki_ymin,ki_ymax], c=valid_col, linestyle='--')
            if threshold is not None:
                ax[0].axvline(threshold, color=test_col, linestyle='--')
                ax[0].axhline(threshold, color=test_col, linestyle='--')
            ax[0].set_title(title, fontdict={'fontsize' : 10})
            
            title = '%s\n%s split %s model on %s features\n%s subset predicted vs actual %s, R^2 = %.3f' % (
                    dataset_name, splitter, model_type, featurizer, subset, "Binding/Inhibition", r2_scores[1])
            bind_ymin = min(min(y_real_bind), min(y_pred_bind)) - 0.1
            bind_ymax = max(max(y_real_bind), max(y_pred_bind)) + 0.1
            ax[1].set_xlim(bind_ymin, bind_ymax)
            ax[1].set_ylim(bind_ymin, bind_ymax)
            ax[1].scatter(y_real_bind, y_pred_bind, s=9, c=train_col, marker='o', alpha=0.4)
            ax[1].set_xlabel('Observed value')
            ax[1].set_ylabel('Predicted value')
            ax[1].plot([bind_ymin,bind_ymax], [bind_ymin,bind_ymax], c=valid_col, linestyle='--')
            if threshold is not None:
                ax[1].axvline(threshold, color=test_col, linestyle='--')
                ax[1].axhline(threshold, color=test_col, linestyle='--')
            ax[1].set_title(title, fontdict={'fontsize' : 10})

    if pdf_dir is not None:
        pdf.close()
        model.log.info("Wrote plot to %s" % pdf_path)


#------------------------------------------------------------------------------------------------------------------------
def plot_pred_vs_actual_from_df(pred_df, actual_col='avg_pIC50_actual', pred_col='avg_pIC50_pred', std_col=None, label=None, ax=None):
    """Plot predicted vs actual values from a trained regression model for a given dataframe.

    Args:
        pred_df (Pandas.DataFrame): A dataframe containing predicted and actual values for each compound.

        actual_col (str): Column with actual values.

        pred_col (str): Column with predicted values.

        label (str): Descriptive label for the plot.

        ax (matplotlib.axes.Axes): Optional, an axes object to plot onto. If None, one is created.

    Returns:
        g (matplotlib.axes.Axes): The axes object with data.

    """
    g=sns.scatterplot(x=actual_col, y=pred_col, data=pred_df, alpha=0.4, ax=ax)
    lims = [
        pred_df[[actual_col,pred_col]].min().min(),  # min of both axes
        pred_df[[actual_col,pred_col]].max().max(),  # max of both axes
    ]
    margin=(lims[1]-lims[0])*0.05
    lims=[lims[0]-margin,lims[1]+margin]
    #g.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    # Draw an identity line
    g.plot(lims, lims, c=test_col, linestyle='--', alpha=0.75, zorder=0)
    # plt.gca().set_aspect('equal', adjustable='box')
    g.set_aspect('equal')
    g.set_xlim(lims)
    g.set_ylim(lims)
    g.set_title(label)
    if std_col is not None:
        filldf=pred_df.copy()
        filldf=filldf.sort_values([actual_col, pred_col])
        g.fill_between(x=filldf[actual_col], y1=filldf[pred_col]-filldf[std_col].abs(), y2=filldf[pred_col]+filldf[std_col].abs(), alpha=0.3, step='mid')
    return g


#------------------------------------------------------------------------------------------------------------------------
def plot_pred_vs_actual_from_file(model_path, external_training_data=None, plot_size=7):
    """Plot predicted vs actual values from a trained regression model from a model tarball. 
    This function only works for locally trained models; otherwise see the `predict_from_model` module.

    Args:
        model_path (str): Path to an AMPL model tar.gz file.
        external_training_data (str): Path to copy of training dataset, if different from path used when model was trained.
        plot_size (float): Height of subplots

    Returns:
        None

    Effects:
        A matplotlib figure is displayed with subplots for each response column and train/valid/test subsets.

    """
    # reload model
    reload_dir = tempfile.mkdtemp()
    with tarfile.open(model_path, mode='r:gz') as tar:
        futils.safe_extract(tar, path=reload_dir)
    
    # reload metadata
    with open(os.path.join(reload_dir, 'model_metadata.json')) as f:
        config=json.loads(f.read())

    if config['model_parameters']['prediction_type']=='classification':
        raise ValueError("plot_pred_vs_actual_from_file() should only be called for regression models. Please try plot_confusion_matrices() instead.")
    
    # load (featurized) data
    dataset_dict=config['training_dataset']
    if external_training_data is not None:
        dataset_dict['dataset_key']=external_training_data
    dataset_key=dataset_dict['dataset_key']
    dataset_name = os.path.splitext(os.path.basename(dataset_key))[0]

    is_featurized=False
    AD_method=None
    uncertainty=config['model_parameters']['uncertainty']
    model_type = config['model_parameters']['model_type']
    featurizer = config['model_parameters']['featurizer']
    if featurizer in ['descriptors','computed_descriptors']:
        desc=config['descriptor_specific']['descriptor_type']
        dataset_key=dataset_key.rsplit('/', maxsplit=1)
        dataset_key=os.path.join(dataset_key[0], 'scaled_descriptors', dataset_key[1].replace('.csv',f'_with_{desc}_descriptors.csv'))
        is_featurized=True
        features_label = f"{desc} descriptors"
    else:
        features_label = f"{featurizer} features"
    # if config['model_parameters']['featurizer'] != 'graphconv':
        # AD_method='z_score'
    df=pd.read_csv(dataset_key)
    
    # reload split file
    dataset_key=dataset_dict['dataset_key']
    split_dict=config['splitting_parameters']
    splitter = split_dict['splitter']
    split_strategy = split_dict['split_strategy']
    if split_strategy == 'k_fold_cv':
        split_file=dataset_key.replace('.csv',f"_{split_dict['num_folds']}_fold_cv_{splitter}_{split_dict['split_uuid']}.csv")
        split_subsets = ['train_valid', 'test']
    else:
        split_file=dataset_key.replace('.csv',f"_{split_strategy}_{splitter}_{split_dict['split_uuid']}.csv")
        split_subsets = ['train', 'valid', 'test']
    split=pd.read_csv(split_file)
    split=split.rename(columns={'cmpd_id':dataset_dict['id_col']})
    
    # merge
    df=df.merge(split, how='left')
    
    # get other values
    response_cols=dataset_dict['response_cols']
    
    # run predictions
    pred_df=pfm.predict_from_model_file(model_path, df, id_col=dataset_dict['id_col'], smiles_col=dataset_dict['smiles_col'], 
                                        response_col=response_cols, is_featurized=is_featurized, AD_method=AD_method, dont_standardize=True)          
    
    # plot
    sns.set_context('notebook')
    nss = len(split_subsets)
    fig, axes = plt.subplots(len(response_cols), nss, figsize=(plot_size*nss, plot_size*len(response_cols)))
    if uncertainty:
        suptitle = f"{dataset_name}  {splitter} {split_strategy} split {model_type} model on {features_label}, predicted vs actual values with uncertainty"
    else:
        suptitle = f"{dataset_name}  {splitter} {split_strategy} split {model_type} model on {features_label}, predicted vs actual values"
    fig.suptitle(suptitle, y=0.95)
    axes = axes.flatten()
    for i,resp in enumerate(response_cols):
        actual_col = f'{resp}_actual'
        pred_col = f'{resp}_pred'
        task_pred_df = pred_df[pred_df[actual_col].notna() & pred_df[pred_col].notna()]
        y_actual = task_pred_df[actual_col].values
        y_pred = task_pred_df[pred_col].values
        if uncertainty:
            std_col = f'{resp}_std'
            y_std = task_pred_df[std_col].values
        else:
            std_col=None
            y_std = 0
        ymin = min(min(y_actual), min(y_pred), min(y_pred-y_std))
        ymax = max(max(y_actual), max(y_pred), max(y_pred+y_std))
        for j, subset in enumerate(split_subsets):
            ax = axes[nss*i + j]
            # Force axes to have same scale for all subsets for same task (but not different tasks!)
            ax.set_xlim(ymin, ymax)
            ax.set_ylim(ymin, ymax)
            tmp = task_pred_df[task_pred_df.subset==subset]
            r2 = metrics.r2_score(tmp[actual_col].values, tmp[pred_col].values)
            ax.set_xlabel(f"Actual {resp}")
            ax.set_ylabel(f"Predicted {resp}")
            # Set subplot title
            if j == 0:
                subtitle = f"{resp} {subset}, {score_type_label['r2']} = {r2:.3f}"
            else:
                subtitle = f"{subset}, {score_type_label['r2']} = {r2:.3f}"
            plot_pred_vs_actual_from_df(tmp, actual_col=actual_col, pred_col=pred_col, std_col = std_col, label=subtitle, ax=ax)


#------------------------------------------------------------------------------------------------------------------------
def plot_perf_vs_epoch(MP, plot_size=7, pdf_dir=None):
    """Plot the current NN model's standard performance metric (r2_score or roc_auc_score) vs epoch number for the training,
    validation and test subsets. If the model was trained with k-fold CV, plot shading for the validation set out to += 1 SD from the mean
    score metric values, and plot the training and test set metrics from the final model retraining rather than the cross-validation
    phase. Make a second plot showing the validation set model choice score used for ranking training epochs and other hyperparameters
    against epoch number.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.
        plot_size (float): Height of subplots

        pdf_dir (str): If given, output the plots to a PDF file in the given directory.

    Returns:
        None

    """
    wrapper = MP.model_wrapper
    if 'train_epoch_perfs' not in wrapper.__dict__:
        raise ValueError("plot_perf_vs_epoch() can only be called for NN models")
    num_epochs = wrapper.num_epochs_trained
    best_epoch = wrapper.best_epoch
    num_folds = len(MP.data.train_valid_dsets)
    if num_folds > 1:
        subset_perf = dict(training = wrapper.train_epoch_perfs[:best_epoch+1], validation = wrapper.valid_epoch_perfs[:num_epochs], 
                        test = wrapper.test_epoch_perfs[:best_epoch+1])
        subset_std = dict(training = wrapper.train_epoch_perf_stds[:best_epoch+1], validation = wrapper.valid_epoch_perf_stds[:num_epochs],
                        test = wrapper.test_epoch_perf_stds[:best_epoch+1])
    else:
        subset_perf = dict(training = wrapper.train_epoch_perfs[:num_epochs], validation = wrapper.valid_epoch_perfs[:num_epochs], 
                        test = wrapper.test_epoch_perfs[:num_epochs])
        subset_std = dict(training = wrapper.train_epoch_perf_stds[:num_epochs], validation = wrapper.valid_epoch_perf_stds[:num_epochs],
                        test = wrapper.test_epoch_perf_stds[:num_epochs])
    model_scores = wrapper.model_choice_scores[:num_epochs]
    model_score_type = MP.params.model_choice_score_type

    if MP.params.prediction_type == 'regression':
        default_score_type = 'r2'
    else:
        default_score_type = 'roc_auc'
    perf_label = score_type_label[default_score_type]
    model_score_type_label = score_type_label[model_score_type]
    num_subplots = 1 + int(model_score_type != default_score_type)

    if MP.params.featurizer in ['descriptors', 'computed_descriptors']:
        features_label = f"{MP.params.descriptor_type} descriptors"
    else:
        features_label = f"{MP.params.featurizer} features"

    if pdf_dir is not None:
        pdf_path = os.path.join(pdf_dir, '%s_perf_vs_epoch.pdf' % os.path.basename(MP.params.output_dir))
        pdf = PdfPages(pdf_path)
    subset_colors = dict(training=train_col, validation=valid_col, test=test_col)
    subset_shades = dict(training=train_shade, validation=valid_shade, test=test_shade)

    with sns.plotting_context("notebook"):
    
        fig, axes = plt.subplots(1, num_subplots, figsize=(plot_size*num_subplots, plot_size))
        """
        suptitle = '%s dataset\n%s vs epoch for %s %s model on %s features with %s split\nBest validation set performance at epoch %d' % (
                MP.params.dataset_name, perf_label, MP.params.model_type,  MP.params.prediction_type,
                MP.params.featurizer,  MP.params.splitter,  best_epoch)
        """
        suptitle = f"{MP.params.dataset_name} dataset, " \
            f"{MP.params.model_type} {MP.params.prediction_type} model on {features_label} with {MP.params.splitter} split\n" \
            f"Best validation set {model_score_type_label} at epoch {best_epoch}"
        fig.suptitle(suptitle, y=0.99)
        # Plot default score type vs epoch
        ax = axes[0] if num_subplots > 1 else axes
        for subset in ['training', 'validation', 'test']:
            epoch = list(range(len(subset_perf[subset])))
            ax.plot(epoch, subset_perf[subset], color=subset_colors[subset], label=subset)
            # Add shading to show variance across folds during cross-validation
            if (num_folds > 1) and (subset == 'validation'):
                ax.fill_between(epoch, subset_perf[subset] + subset_std[subset], subset_perf[subset] - subset_std[subset],
                                alpha=0.3, facecolor=subset_shades[subset], linewidth=0)
        ax.axvline(best_epoch, color=test_col, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(perf_label)
        title = f"{perf_label} vs epoch"
        #if num_subplots == 1:
        #    title += f", Best validation set {perf_label} at epoch {best_epoch)"
        ax.set_title(title, fontdict={'fontsize' : 12})
        legend = ax.legend(loc='lower right')
    
        # Now plot the score used for choosing the best epoch and model params, if different from the default R2 or ROC AUC
        if num_subplots == 2:
            ax = axes[1]
            """
            title = '%s dataset\n%s vs epoch for %s %s model on %s features with %s split\nBest validation set performance at epoch %d' % (
                    MP.params.dataset_name, model_score_type, MP.params.model_type,  MP.params.prediction_type,
                    MP.params.featurizer,  MP.params.splitter,  best_epoch)
            """
            epoch = list(range(num_epochs))
            ax.plot(epoch, model_scores, color=subset_colors['validation'])
            plt.axvline(best_epoch, color=test_col, linestyle='--')
            ax.set_xlabel('Epoch')
            if model_score_type in perf.loss_funcs:
                score_label = f"Negative {model_score_type_label}"
                title = f"Validation set negative {model_score_type_label} vs epoch"
            else:
                score_label = model_score_type_label
                title = f"Validation set {model_score_type_label} vs epoch"
            ax.set_ylabel(score_label)
            ax.set_title(title, fontdict={'fontsize' : 12})
    if pdf_dir is not None:
        pdf.savefig(fig)
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)



#------------------------------------------------------------------------------------------------------------------------
def _get_perf_curve_data(MP, epoch_label, curve_type='ROC'):
    """Common code for ROC and precision-recall curves. Returns true classes and active class probabilities
    for each training/test data subset.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.

        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.

        threshold (float): Threshold activity value to mark on plot with dashed lines.

        error_bars (bool): If true and if uncertainty estimates are included in the model predictions, draw error bars
            at +- 1 SD from the predicted y values.

        pdf_dir (str): If given, output the plots to a PDF file in the given directory.

    Returns:
        None

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
def get_classifier_perf_data_from_pipeline(MP, epoch_label='best'):
    """Retrieve predicted and true classes, class probabilities and model metrics from a classification model trained
    in the current Python session.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
    
    Returns:
        (dict): A dictionary of dictionaries, keyed by split subset, containing the true classes, predicted classes,
        and predicted class probabilities for each compound in each subset, along with a subdictionary of metrics for each task/subset.
    """
    if MP.params.prediction_type != 'classification':
        MP.log.error("get_classifier_perf_data_from_pipeline can only be called for classification models")
        return {}

    if MP.run_mode == 'training':
        subsets = ['train', 'valid', 'test']
    else:
        subsets = ['full']
    wrapper = MP.model_wrapper
    classif_data = {}
    tasks = MP.params.response_cols
    ntasks = len(tasks)
    for itask, task in enumerate(tasks):
        classif_data[task] = {}
    training_metrics = get_metrics_from_metadata(MP.model_metadata)
    for subset in subsets:
        perf_data = wrapper.get_perf_data(subset, epoch_label)
        true_classes = perf_data.get_real_values()
        ids, pred_classes, class_probs, prob_stds = perf_data.get_pred_values()
        nclasses = class_probs.shape[-1]
        for itask, task in enumerate(tasks):
            if nclasses == 2:
                # Only return class 1 probabilities for binary classifiers
                task_class_probs = class_probs[:,itask,1]
            else:
                task_class_probs = class_probs[:,itask,:]
            classif_data[task][subset] = dict(true_class=true_classes[:,itask], pred_class=pred_classes[:,itask], class_probs=task_class_probs,
                                              metrics=training_metrics[task][subset])
    return classif_data


#------------------------------------------------------------------------------------------------------------------------
def get_metrics_from_metadata(metadata_dict, epoch_label='best'):
    """Retrieve model performance metrics from a model metadata dictionary.
    
    Args:
        metadata_dict (dict): A model metadata structure, which can be either
        from a saved model or a model trained in the current Python session.
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
    
    Returns:
        (dict): A nested dictionary containing the metric values, keyed first by task, then by subset and finally by metric type.
    """
    training_metrics = metadata_dict['training_metrics']
    prediction_type = metadata_dict['model_parameters']['prediction_type']
    metric_keys = dict(
        classification=dict(
            singletask = ['roc_auc_score', 'prc_auc_score', 'accuracy_score', 'precision', 'recall_score', 'bal_accuracy',
                          'npv', 'cross_entropy', 'kappa', 'matthews_cc', 'confusion_matrix'],
            multitask = ['task_roc_auc_scores', 'task_prc_auc_scores', 'task_accuracies', 'task_precisions', 'task_recalls',
                         'task_bal_accuracies', 'task_npvs', 'task_cross_entropies', 'task_kappas', 'task_matthews_ccs', 'confusion_matrix'],
            names = ['roc_auc', 'prc_auc', 'accuracy', 'precision', 'recall', 'bal_accuracy', 'npv', 'cross_entropy', 'kappa', 'MCC',
                     'confusion_matrix']
        ),
        regression=dict(
            singletask = ['r2_score', 'mae_score', 'rms_score'],
            multitask = ['task_r2_scores', 'task_mae_scores', 'task_rms_scores'],
            names = ['r2', 'mae', 'rmse']
        )
    )
    tasks = metadata_dict['training_dataset']['response_cols']
    ntasks = len(tasks)
    if ntasks == 1:
        metric_dict = dict(zip(metric_keys[prediction_type]['names'], metric_keys[prediction_type]['singletask']))
    else:
        metric_dict = dict(zip(metric_keys[prediction_type]['names'], metric_keys[prediction_type]['multitask']))
    perf_metrics = {}
    for task in tasks:
        perf_metrics[task] = {}
    for ss_metrics in training_metrics:
        if ss_metrics['label'] == epoch_label:
            subset = ss_metrics['subset']
            pred_results = ss_metrics['prediction_results']
            # Iterate over tasks and metric types, get values from task_precisions etc. if ntasks > 1, otherwise from precision etc.
            for itask, task in enumerate(tasks):
                perf_metrics[task][subset] = {}
                for metric_name, metric_key in metric_dict.items():
                    if ntasks == 1:
                        perf_metrics[task][subset][metric_name] = pred_results[metric_key]
                    else:
                        perf_metrics[task][subset][metric_name] = pred_results[metric_key][itask]
    return perf_metrics

#------------------------------------------------------------------------------------------------------------------------
def get_metrics_from_model_pipeline(pipeline, epoch_label='best'):
    """Retrieve model performance metrics from a ModelPipeline object.
    
    Args:
        pipeline (ModelPipeline): A ModelPipeline for a model trained in the current Python session.
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
    
    Returns:
        (dict): A nested dictionary containing the metric values, keyed first by task, then by subset and finally by metric type.
    """
    return get_metrics_from_metadata(pipeline.model_metadata, epoch_label=epoch_label)


#------------------------------------------------------------------------------------------------------------------------
def get_metrics_from_model_file(model_path, epoch_label='best'):
    """Retrieve model performance metrics from a trained model saved in the local filesystem.
    
    Args:
        model_path (str): Path to a saved model .tar.gz file or directory
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
    
    Returns:
        (dict): A nested dictionary containing the metric values, keyed first by task, then by subset and finally by metric type.
    """
    model_reader = mfr.ModelFileReader(model_path)
    return get_metrics_from_metadata(model_reader.metadata_dict, epoch_label=epoch_label)


#------------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrices(model, epoch_label='best', plot_size=7):
    """Displays the confusion matrix for each split subset for a classification model.

    Args:
        model (ModelPipeline or str): A classification model. The model may be represented by either a ModelPipeline object 
        or a file path to a saved model .tar.gz file or directory.
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
        plot_size (float): Height of subplots.
    
    Returns:
        None
    """
    if isinstance(model, mp.ModelPipeline):
        metrics_dict = get_metrics_from_model_pipeline(model, epoch_label)
    elif isinstance(model, str):
        metrics_dict = get_metrics_from_model_file(model, epoch_label)
    else:
        raise ValueError('model must be either a ModelPipeline or a path to a saved model')
    tasks = list(metrics_dict.keys())
    subsets = list(metrics_dict[tasks[0]].keys())
    with sns.plotting_context('poster'):
        fig, axes = plt.subplots(len(tasks), len(subsets), figsize=(plot_size*len(subsets), plot_size*len(tasks)))
        axes = axes.flatten()
        for it, task in enumerate(tasks):
            for iss, subset in enumerate(subsets):
                if len(tasks)>1:
                    cmatrix = np.array(metrics_dict[task][subset]['confusion_matrix'])
                else:
                    cmatrix = np.array(metrics_dict[task][subset]['confusion_matrix'][0])
                cmd = ConfusionMatrixDisplay(cmatrix)
                ax = axes[len(subsets)*it + iss]
                cmd.plot(ax=ax, colorbar=False)
                ax.set_title(f"{task}, {subset} subset")
                ax.set_ylabel("True class")
                ax.set_xlabel("Predicted class")
                plt.tight_layout()


#------------------------------------------------------------------------------------------------------------------------
def plot_model_metrics(model, epoch_label='best', plot_size=7):
    """Displays a bar plot of metrics for each split subset for a classification or regression model, for each task.

    Args:
        model (ModelPipeline or str): A model. The model may be represented by either a ModelPipeline object 
        or a file path to a saved model .tar.gz file or directory.
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
        plot_size (float): Height of subplots.
    
    Returns:
        None
    """
    # Save current color palette and restore it later
    old_palette = sns.color_palette()
    sns.set_palette('colorblind')

    if isinstance(model, mp.ModelPipeline):
        metrics_dict = get_metrics_from_model_pipeline(model, epoch_label)
    elif isinstance(model, str):
        metrics_dict = get_metrics_from_model_file(model, epoch_label)
    else:
        raise ValueError('model must be either a ModelPipeline or a path to a saved model')
    tasks = list(metrics_dict.keys())
    subsets = list(metrics_dict[tasks[0]].keys())
    metric_vars = [key for key in metrics_dict[tasks[0]][subsets[0]].keys() if key != 'confusion_matrix']
    task_list = []
    subset_list = []
    metric_list = []
    value_list = []
    for it, task in enumerate(tasks):
        for iss, subset in enumerate(subsets):
            tsm_dict = metrics_dict[task][subset]
            for mvar in metric_vars:
                task_list.append(task)
                subset_list.append(subset)
                metric_list.append(score_type_label[mvar])
                value_list.append(tsm_dict[mvar])
    metric_df = pd.DataFrame(dict(task=task_list, subset=subset_list, metric=metric_list, value=value_list))
    with sns.plotting_context('poster'):
        fgrid = sns.FacetGrid(data=metric_df, row='task', col='subset', height=plot_size, hue='metric', sharex=True, sharey=True)
        fgrid.map_dataframe(sns.barplot, x='value', y='metric')

    # Restore previous matplotlib color cycle
    sns.set_palette(old_palette)

#------------------------------------------------------------------------------------------------------------------------
def plot_ROC_curve(MP, epoch_label='best', plot_size=7, pdf_dir=None):
    """Plot ROC curves for a classification model.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.

        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.

        pdf_dir (str): If given, output the plots to a PDF file in the given directory.

    Returns:
        None

    """
    params = MP.params
    curve_data = _get_perf_curve_data(MP, epoch_label, 'ROC')
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
    subset_colors = dict(train=train_col, valid=valid_col, test=test_col, full=full_col)
    # For multitask, do a separate figure for each task
    ntasks = curve_data[subsets[0]]['prob_active'].shape[1]
    with sns.plotting_context('talk'):
        for i in range(ntasks):
            fig, ax = plt.subplots(figsize=(plot_size,plot_size))
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
def plot_prec_recall_curve(MP, epoch_label='best', plot_size=7, pdf_dir=None):
    """Plot precision-recall curves for a classification model.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.
        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.
        plot_size (float): Height of subplots
        pdf_dir (str): If given, output the plots to a PDF file in the given directory.

    Returns:
        None

    """
    params = MP.params
    curve_data = get_classifier_perf_data_from_pipeline(MP, epoch_label=epoch_label)
    tasks = list(curve_data.keys())
    ntasks = len(tasks)
    with sns.plotting_context('notebook'):
        fig, axes = plt.subplots(ntasks, 1, figsize=(plot_size, plot_size*ntasks))
        subset_colors = dict(train=train_col, valid=valid_col, test=test_col, full=full_col)
        for itt, task in enumerate(tasks):
            if ntasks > 1:
                ax = axes[itt]
            else:
                ax = axes
            subsets = list(curve_data[task].keys())
            for iss, subset in enumerate(subsets):
                ss_data = curve_data[task][subset]
                prd = PrecisionRecallDisplay.from_predictions(ss_data['true_class'], ss_data['class_probs'],
                                                            ax=ax, drawstyle='default', c=subset_colors[subset], name=subset)
                ax.set_title(f"Response column: '{task}'")    
            legend = ax.legend(loc='lower left')

    if pdf_dir is not None:
        pdf.savefig(fig)
    if pdf_dir is not None:
        pdf.close()
        MP.log.info("Wrote plot to %s" % pdf_path)

#------------------------------------------------------------------------------------------------------------------------
def old_plot_prec_recall_curve(MP, epoch_label='best', plot_size=7, pdf_dir=None):
    """Plot precision-recall curves for a classification model.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.

        epoch_label (str): Label for training epoch to draw predicted values from. Currently 'best' is the only allowed value.

        pdf_dir (str): If given, output the plots to a PDF file in the given directory.

    Returns:
        None

    """
    params = MP.params
    curve_data = _get_perf_curve_data(MP, epoch_label, 'precision-recall')
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
    subset_colors = dict(train=train_col, valid=valid_col, test=test_col, full=full_col)
    # For multitask, do a separate figure for each task
    ntasks = curve_data[subsets[0]]['prob_active'].shape[1]
    for i in range(ntasks):
        fig, ax = plt.subplots(figsize=(plot_size,plot_size))
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
    """Projects features of a model's input dataset using UMAP to 2D or 3D coordinates and draws a scatterplot.
    Shape-codes plot markers to indicate whether the associated compound was in the training, validation or
    test set. For classification models, also uses the marker shape to indicate whether the compound's class was correctly
    predicted, and uses color to indicate whether the true class was active or inactive. For regression models, uses
    the marker color to indicate the discrepancy between the predicted and actual values.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.

        ndim (int): Number of dimensions (2 or 3) to project features into.

        num_neighbors (int): Number of nearest neighbors used by UMAP for manifold approximation.
            Larger values give a more global view of the data, while smaller values preserve more local detail.

        min_dist (float): Parameter used by UMAP to set minimum distance between projected points.

        fit_to_train (bool): If true (the default), fit the UMAP projection to the training set feature vectors only.
            Otherwise, fit it to the entire dataset.

        dist_metric (str): Name of metric to use for initial distance matrix computation. Check UMAP documentation
            for supported values. The metric should be appropriate for the type of features used in the model (fingerprints
            or descriptors); note that `jaccard` is equivalent to Tanimoto distance for ECFP fingerprints.

        dist_metric_kwds (dict): Additional key-value pairs used to parameterize dist_metric; see the UMAP documentation.
            In particular, dist_metric_kwds['p'] specifies the power/exponent for the Minkowski metric.

        target_weight (float): Weighting factor determining balance between activities and feature values in determining topology
            of projected points. A weight of zero prioritizes the feature vectors; weight = 1 prioritizes the activity values,
            so that compounds with the same activity tend to be clustered together.

        random_seed (int): Seed for random number generator.

        pdf_dir (str): If given, output the plot to a PDF file in the given directory.

    Returns:
        None

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
            marker_map = {'training/correct' : 'o', 'training/incorrect' : 's', 
                        'valid/correct' : '^', 'valid/incorrect' : 'v', 
                        'test/correct' : 'P', 'test/incorrect' : '*'}
            size_map = {'training/correct' : 49, 'training/incorrect' : 64, 
                        'valid/correct' : 81, 'valid/incorrect' : 125, 
                        'test/correct' : 125, 'test/incorrect' : 225}
            if ndim == 2:
                ax = fig.add_subplot(111)
                sns.scatterplot(x='umap_X', y='umap_Y', hue='actual', palette=binary_pal, 
                        style='subset', markers=marker_map,
                        style_order=['training/correct', 'training/incorrect', 'valid/correct', 'valid/incorrect',
                                     'test/correct', 'test/incorrect' ],
                        size='subset', sizes=size_map,
                        data=proj_df, ax=ax)
            elif ndim == 3:
                ax = fig.add_subplot(111, projection='3d')
                colors = [binary_pal[a] for a in proj_df.actual.values]
                markers = [marker_map[subset] for subset in proj_df.subset.values]
                #ax.scatter(proj_df.umap_X.values, proj_df.umap_Y.values, proj_df.umap_Z.values, c=colors, m=markers, s=49)
                ax.scatter(proj_df.umap_X.values, proj_df.umap_Y.values, proj_df.umap_Z.values, c=colors, s=49)
        else:
            # regression model
            proj_df['subset'] = dset_subset
            proj_df['error'] = preds[:,i] - all_actual[:,i]
            marker_map = {'training' : 'o', 'valid' : 'v', 'test' : 'P'}
            ncol = 12
            if ndim == 2:
                ax = fig.add_subplot(111)
                #sns.scatterplot(x='umap_X', y='umap_Y', hue='error',
                sns.scatterplot(x='umap_X', y='umap_Y', hue='error', palette=continuous_pal, 
                        size='actual', sizes=(49,144), alpha=0.95,
                        style='subset', markers=marker_map, style_order=['training', 'valid', 'test'],
                        data=proj_df, ax=ax)
            elif ndim == 3:
                ax = fig.add_subplot(111, projection='3d')
                # Map errors to palette indices
                errs = proj_df.error.values.astype(np.int32)
                ind = 1 + errs - min(errs)
                ind[ind >= ncol] = ncol-1
                colors = [continuous_pal[i] for i in ind]
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
    """Project features of whole dataset to 2 dimensions, without regard to response values. Plot training & validation set
    or training and test set compounds, color- and symbol-coded according to actual classification and split set.
    The plot does not take predicted values into account at all. Does not work with regression data.

    Args:
        MP (`ModelPipeline`): Pipeline object for a model that was trained in the current Python session.

        num_neighbors (int): Number of nearest neighbors used by UMAP for manifold approximation.
            Larger values give a more global view of the data, while smaller values preserve more local detail.

        min_dist (float): Parameter used by UMAP to set minimum distance between projected points.

        dist_metric (str): Name of metric to use for initial distance matrix computation. Check UMAP documentation
            for supported values. The metric should be appropriate for the type of features used in the model (fingerprints
            or descriptors); note that `jaccard` is equivalent to Tanimoto distance for ECFP fingerprints.

        dist_metric_kwds (dict): Additional key-value pairs used to parameterize dist_metric; see the UMAP documentation.
            In particular, dist_metric_kwds['p'] specifies the power/exponent for the Minkowski metric.

        random_seed (int): Seed for random number generator.

        pdf_dir (str): If given, output the plot to a PDF file in the given directory.


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
                pal = {'train/active' : train_active_col, 'train/inactive' : train_inactive_col, 
                            'valid/active' : test_active_col, 'valid/inactive' : test_inactive_col}
                marker_map = {'train/active' : 'o', 'train/inactive' : 's', 
                            'valid/active' : '*', 'valid/inactive' : 'v'}
                size_map = {'train/active' : 64, 'train/inactive' : 49, 
                            'valid/active' : 192, 'valid/inactive' : 81}
                style_order=['train/inactive', 'valid/inactive', 'train/active', 'valid/active' ]
            else:
                pal = {'train/active' : train_active_col, 'train/inactive' : train_inactive_col, 
                            'test/active' : test_active_col, 'test/inactive' : test_inactive_col}
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
