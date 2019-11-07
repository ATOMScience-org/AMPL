"""
Plotting routines for visualizing performance of models from hyperparameter search
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


#matplotlib.style.use('ggplot')
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize=12)
sns.set(font_scale=1.1)


#------------------------------------------------------------------------------------------------------------------------
def hyper_perf_plots(collection='pilot_fixed', pred_type='regression', top_n=1, subset='test'):
    """
    Get results for models in the given collection. For each training dataset, separate the
    results by model type and featurizer, and make a scatter plot of <subset> set r2_score vs
    model type/featurizer combo. Then make plots of r2 vs model parameters relevant for each
    model type.
    """
    
    res_dir = '/usr/local/data/%s_perf' % collection
    plt_dir = '%s/Plots' % res_dir
    os.makedirs(plt_dir, exist_ok=True)
    res_files = os.listdir(res_dir)
    suffix = '_%s_model_perf_metrics.csv' % collection
    feat_marker_map = {'descriptors' : 'o', 'ecfp' : 's', 'graphconv' : 'P'}
    feat_model_marker_map = {'NN/descriptors' : 'o', 'NN/ecfp' : 's', 'RF/descriptors' : 'P', 'RF/ecfp': 'X'}

    if pred_type == 'regression':
        metric_type = 'r2_score'
    else:
        metric_type = 'roc_auc_score'
    # TODO: Make these generic
    green_red_pal = sns.blend_palette(['red', 'green'], 12, as_cmap=True)
    mfe_pal = {'32/500' : 'red', '32/499' : 'blue', '63/300' : 'forestgreen', '63/499' : 'magenta'}
    res_data = []
    # Plot score vs learning rate for NN models
    pdf_path = '%s/%s_%s_NN_perf_vs_learning_rate.pdf' % (plt_dir, collection, subset)
    pdf = PdfPages(pdf_path)
    for res_file in res_files:
        try:
            if not res_file.endswith(suffix):
                print("File {} doesn't end with proper suffix".format(res_file))
                continue
            dset_name = res_file.replace(suffix, '')
            res_path = os.path.join(res_dir, res_file)
            res_df = pd.read_csv(res_path, index_col=False)
            res_df['combo'] = ['%s/%s' % (m,f) for m, f in zip(res_df.model_type.values, res_df.featurizer.values)]
            res_data.append(res_df)
            nn_df = res_df[res_df.model_type == 'NN']
            rf_df = res_df[res_df.model_type == 'RF']
            fig = plt.figure(figsize=(10,13))
            axes = fig.subplots(2,1,sharex=True)
            ax1 = axes[0]
            ax1.set_title('%s NN model perf vs learning rate' % dset_name, fontdict={'fontsize' : 12})
            ax1.set_xscale('log')
            sns.scatterplot(x='learning_rate', y='{0}_{1}'.format(metric_type, subset), hue='best_epoch', palette=green_red_pal,
                            style='featurizer', markers=feat_marker_map, data=nn_df, ax=ax1)
            ax2 = axes[1]
            ax2.set_xscale('log')
            sns.scatterplot(x='learning_rate', y='best_epoch', hue='best_epoch', palette=green_red_pal,
                            style='featurizer', markers=feat_marker_map, data=nn_df, ax=ax2)
            pdf.savefig(fig)
        except:
            continue

    pdf.close()
    print("Wrote plots to %s" % pdf_path)
    # Plot score vs max depth for RF models
    pdf_path = '%s/%s_%s_RF_perf_vs_max_depth.pdf' % (plt_dir, collection, subset)
    pdf = PdfPages(pdf_path)
    for res_file in res_files:
        try:
            if not res_file.endswith(suffix):
                continue
            dset_name = res_file.replace(suffix, '')
            res_path = os.path.join(res_dir, res_file)
            res_df = pd.read_csv(res_path, index_col=False)
            res_df['combo'] = ['%s/%s' % (m,f) for m, f in zip(res_df.model_type.values, res_df.featurizer.values)]
            res_df['dataset'] = dset_name
            res_df = res_df.sort_values('{0}_{1}'.format(metric_type, subset), ascending=False)
            rf_df = res_df[res_df.model_type == 'RF']
            fig = plt.figure(figsize=(8,8))
            ax1 = fig.add_subplot(111)
            rf_df['max_feat/estimators'] = ['%d/%d' % (mf,est) for mf,est in zip(rf_df.rf_max_features.values, rf_df.rf_estimators.values)]
            ax1.set_title('%s RF model perf vs max depth' % dset_name, fontdict={'fontsize' : 12})
            sns.scatterplot(x='rf_max_depth', y='{0}_{1}'.format(metric_type, subset), hue='max_feat/estimators', palette=mfe_pal, style='featurizer', markers=feat_marker_map, data=rf_df, ax=ax1)
            pdf.savefig(fig)
        except:
            continue

    pdf.close()
    print("Wrote plots to %s" % pdf_path)

    # Plot score vs model type and featurizer. Compile a table of top scoring model type/featurizers for each dataset.
    pdf_path = '%s/%s_%s_perf_vs_model_type_featurizer.pdf' % (plt_dir, collection, subset)
    pdf = PdfPages(pdf_path)

    datasets = []
    top_model_feat = []
    top_scores = []
    top_grouped_models = []
    num_samples = []
    top_combo_dsets = []
    for res_file in res_files:
        try:
            if not res_file.endswith(suffix):
                continue
            dset_name = res_file.replace(suffix, '')
            datasets.append(dset_name)
            res_df['dataset'] = dset_name
            print(dset_name)
            res_df = res_df.sort_values('{0}_{1}'.format(metric_type, subset), ascending=False)
            res_path = os.path.join(res_dir, res_file)
            res_df = pd.read_csv(res_path, index_col=False)
            res_df['model_type/feat'] = ['%s/%s' % (m,f) for m, f in zip(res_df.model_type.values, res_df.featurizer.values)]
            res_df = res_df.sort_values('{0}_{1}'.format(metric_type, subset), ascending=False)
            grouped_df = res_df.groupby('model_type/feat').apply(
                lambda t: t.head(top_n)
            ).reset_index(drop=True)
            top_grouped_models.append(grouped_df)
            top_combo = res_df['model_type/feat'].values[0]
            top_combo_dsets.append(top_combo + dset_name.lstrip('ATOM_GSK_dskey'))
            top_score = res_df['{0}_{1}'.format(metric_type, subset)].values[0]
            top_model_feat.append(top_combo)
            top_scores.append(top_score)
            # Need to make sure which method works
            #num_samples.append(res_df['Dataset Size'][0])
            print(res_df['num_samples'])
            num_samples.append(res_df['num_samples'])
            fig = plt.figure(figsize=(8,10))
            ax = fig.add_subplot(111)
            ax.set_title('%s %s set perf vs model type and features' % (dset_name, subset), fontdict={'fontsize' : 10})
            sns.scatterplot(x='model_type/feat', y='{0}_{1}'.format(metric_type, subset),
                            style='featurizer', markers=feat_marker_map, data=res_df, ax=ax)
            pdf.savefig(fig)
        except Exception as e:
            print("Couldn't get top combo")
            print(e)
            continue
    pdf.close()
    # Plot score vs model type and featurizer. Compile a table of top scoring model type/featurizers for each dataset.
    try:
        top_score_model_df = pd.DataFrame({'Dataset' : datasets, 'Top scoring model/features' : top_model_feat,
                                           '{0} set {1}'.format(subset, metric_type) : top_scores,
                                           'Dataset Size': num_samples, 'top_combo_dsets': top_combo_dsets})
        top_score_model_df.to_csv('%s/%s_%s_top_scoring_models.csv' % (res_dir, collection, subset), index=False)
        #TODO: working on adding in num_samples vs top r^2 plot
        pdf_path = '%s/%s_%s_perf_vs_num_samples.pdf' % (plt_dir, collection, subset)
        pdf = PdfPages(pdf_path)
        fig = plt.figure(figsize=(10,9))
        ax = fig.add_subplot(111)
        ax.set_title('%s set perf vs number of compounds' % subset, fontdict={'fontsize' : 10})
        # sns.scatterplot(x='Dataset Size', y='{0} set {1}'.format(subset, metric_type), style='top_combo_dsets',
        #                data=top_score_model_df, ax=ax)
        sns.scatterplot(x='Dataset Size', y='{0} set {1}'.format(subset, metric_type), data=top_score_model_df, ax=ax)
        pdf.savefig(fig)
        pdf.close()
    except Exception as e:
        print(e)
        print("Can't get num samples info")
    try:
        combined_df = pd.concat(res_data, ignore_index=True)
        combined_file = '%s/%s_%s_combined_data.csv' % (res_dir, collection, subset)
        combined_df.to_csv(combined_file, index=False)
        print("Wrote combined results table to %s" % combined_file)
    except:
        print("Can't get combined results")
    try:
        combined_df = pd.concat(top_grouped_models, ignore_index=True)
        combined_df.to_csv('%s/%s_%s_top_scoring_models_grouped.csv' % (res_dir, collection, subset), index=False)
        print("Wrote top grouped results table")
    except:
        print("Can't get grouped results")

    
#------------------------------------------------------------------------------------------------------------------------
def hyper_perf_plot_file(dset_name, col_name='pilot_fixed', pred_type='regression', top_n=1, subset='test'):
    """
    Get results for models in the given collection. For each training dataset, separate the
    results by model type and featurizer, and make a scatter plot of <subset> set r2_score vs
    model type/featurizer combo. Then make plots of r2 vs model parameters relevant for each
    model type.
    """
    res_dir = '/usr/local/data/%s_perf' % col_name
    plt_dir = '%s/Plots' % res_dir
    os.makedirs(plt_dir, exist_ok=True)
    feat_marker_map = {'descriptors' : 'o', 'ecfp' : 's', 'graphconv' : 'P'}
    if pred_type == 'regression':
        metric_type = 'r2_score'
    else:
        metric_type = 'roc_auc_score'
    # TODO: Make these generic
    green_red_pal = sns.blend_palette(['red', 'green'], 12, as_cmap=True)
    mfe_pal = {'32/500' : 'red', '32/499' : 'blue', '63/300' : 'forestgreen', '63/499' : 'magenta'}
    suffix = '_%s_model_perf_metrics.csv' % col_name

    print(dset_name)
    res_file = '%s_%s_model_perf_metrics.csv' % (dset_name, col_name)
    res_path = os.path.join(res_dir, res_file)
    res_df = pd.read_csv(res_path, index_col=False)
    print(res_df.keys())
    res_df['combo'] = ['%s/%s' % (m,f) for m, f in zip(res_df.model_type.values, res_df.featurizer.values)]
    res_df['dataset'] = dset_name
    res_df = res_df.sort_values('{0}_{1}'.format(metric_type, subset), ascending=False)
    nn_df = res_df[res_df.model_type == 'NN']
    rf_df = res_df[res_df.model_type == 'RF']
    # Plot score vs learning rate for NN models
    
    pdf_path = '%s/%s_%s_%s_NN_perf_vs_learning_rate.pdf' % (plt_dir, col_name, dset_name, subset)
    pdf = PdfPages(pdf_path)
    try:
        fig = plt.figure(figsize=(10,13))
        axes = fig.subplots(2,1,sharex=True)
        ax1 = axes[0]
        ax1.set_title('%s NN model perf vs learning rate' % dset_name, fontdict={'fontsize' : 12})
        ax1.set_xscale('log')
        sns.scatterplot(x='learning_rate', y='{0}_{1}'.format(metric_type, subset), hue='best_epoch', palette=green_red_pal, style='featurizer', markers=feat_marker_map, data=nn_df, ax=ax1)
        ax2 = axes[1]
        ax2.set_xscale('log')
        sns.scatterplot(x='learning_rate', y='best_epoch', hue='best_epoch', palette=green_red_pal,
                        style='featurizer', markers=feat_marker_map, data=nn_df, ax=ax2)
        pdf.savefig(fig)
    except:
        return

    pdf.close()
    print("Wrote plots to %s" % pdf_path)

    # Plot score vs max depth for RF models
    pdf_path = '%s/%s_%s_%s_RF_perf_vs_max_depth.pdf' % (plt_dir, col_name, dset_name, subset)
    pdf = PdfPages(pdf_path)

    try:
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)
        rf_df['max_feat/estimators'] = ['%d/%d' % (mf,est) for mf,est in zip(rf_df.rf_max_features.values, rf_df.rf_estimators.values)]
        ax1.set_title('%s RF model perf vs max depth' % dset_name, fontdict={'fontsize' : 12})
        sns.scatterplot(x='rf_max_depth', y='{0}_{1}'.format(metric_type, subset), hue='max_feat/estimators', palette=mfe_pal, style='featurizer', markers=feat_marker_map, data=rf_df, ax=ax1)

        pdf.savefig(fig)
    except:
        return

    pdf.close()
    print("Wrote plots to %s" % pdf_path)

    # Plot score vs model type and featurizer.
    pdf_path = '%s/%s_%s_%s_perf_vs_model_type_featurizer.pdf' % (plt_dir, col_name, dset_name, subset)
    pdf = PdfPages(pdf_path)
    try:
        res_df = pd.read_csv(res_path, index_col=False)
        res_df['model_type/feat'] = ['%s/%s' % (m,f) for m, f in zip(res_df.model_type.values, res_df.featurizer.values)]
        res_df = res_df.sort_values('{0}_{1}'.format(metric_type, subset), ascending=False)
        fig = plt.figure(figsize=(8,10))
        ax = fig.add_subplot(111)
        ax.set_title('%s %s set perf vs model type and features' % (dset_name, subset), fontdict={'fontsize' : 10})
        sns.scatterplot(x='model_type/feat', y='{0}_{1}'.format(metric_type, subset),
                        style='featurizer', markers=feat_marker_map, data=res_df, ax=ax)
        pdf.savefig(fig)
    except:
        return
    pdf.close()
    
    #Compile a table of top scoring model type/featurizers for each dataset.
    try:
        top_model_feat = []
        top_scores = []
        print(res_df.shape)
        grouped_df = res_df.groupby('model_type/feat').apply(
            lambda t: t.head(top_n)
        ).reset_index(drop=True)
        grouped_dir = '%s/%s_%s_%s_top_scoring_models.csv' % (res_dir, col_name, dset_name, subset)
        grouped_df.to_csv(grouped_dir, index=False)
        print("Wrote top results table to %s" % grouped_dir)
    except:
        return

# ------------------------------------------------------------------------------------------------------------------------
def plot_uncertainties(results_df, col_name='pilot'):
    """
    Get results for models in the given collection. For each training dataset, separate the
    results by model type and featurizer, and make a scatter plot of <subset> set r2_score vs
    model type/featurizer combo. Then make plots of r2 vs model parameters relevant for each
    model type.
    """
    res_dir = '/ds/projdata/gsk_data/model_analysis/'
    plt_dir = '%s/Plots' % res_dir
    os.makedirs(plt_dir, exist_ok=True)
    '''
    res_file = '%s_uncertainties.csv' % (col_name)
    res_path = os.path.join(res_dir, res_file)
    results_df = pd.read_csv(res_path, index_col=False)
    '''
    cats = results_df['dset_key'].unique()
    cats.sort()
    sns.set(font_scale=1.5)
    model_type = results_df.model_type.values[0]
    splitter = results_df.splitter.values[0]
    plt_dir = '/usr/local/data/ddm_pipeline_paper/'
    pdf_path = os.path.join(plt_dir, '%s_uncertainty_vs_error.pdf' % col_name)
    pdf = PdfPages(pdf_path)
    for dset_name in cats:
        print(dset_name)
        try:
            fig = plt.figure(figsize=(10, 13))
            ax = fig.add_subplot(111)
            title_name = dset_name.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace(
                'HEK_IonWorks_Electrophys_2Hz_', '').rstrip('.csv').replace('_', ' ')
            ax.set_title('%s std vs error for %s and %s split' % (title_name,
                                                                  model_type,
                                                                  splitter))
            data = results_df[results_df.dset_key == dset_name]
            std_col = [col for col in results_df.columns if 'std' in col][0]
            actual_col = [col for col in results_df.columns if 'actual' in col][0]
            g = sns.scatterplot(x="error", y=std_col, hue=actual_col, palette="YlOrRd", data=data, ax=ax)
            ax.set(xlabel='Prediction error', ylabel='Uncertainty of prediction')
            pdf.savefig(fig)
        except Exception as e:
            print(e)
            continue
    
    pdf.close()
    
    pdf_path = '%s/%s_uncertainty_vs_pred.pdf' % (plt_dir, col_name)
    pdf = PdfPages(pdf_path)
    for dset_name in cats:
        print(dset_name)
        try:
            fig = plt.figure(figsize=(10, 13))
            ax = fig.add_subplot(111)
            title_name = dset_name.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace(
                'HEK_IonWorks_Electrophys_2Hz_', '').rstrip('.csv').replace('_', ' ')
            ax.set_title('%s std vs error for %s and %s split' % (title_name,
                                                                  model_type,
                                                                  splitter))
            data = results_df[results_df.dset_key == dset_name]
            std_col = [col for col in results_df.columns if 'std' in col][0]
            actual_col = [col for col in results_df.columns if 'actual' in col][0]
            g = sns.scatterplot(x="error", y='pred', hue=actual_col, palette="YlOrRd", data=data, ax=ax)
            ax.set(xlabel='Prediction error', ylabel='Predicted value')
            pdf.savefig(fig)
        except Exception as e:
            print(e)
            continue
    
    pdf.close()
    
    pdf_path = '%s/%s_uncertainty_vs_actual.pdf' % (plt_dir, col_name)
    pdf = PdfPages(pdf_path)
    for dset_name in cats:
        print(dset_name)
        try:
            fig = plt.figure(figsize=(10, 13))
            ax = fig.add_subplot(111)
            title_name = dset_name.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace(
                'HEK_IonWorks_Electrophys_2Hz_', '').rstrip('.csv').replace('_', ' ')
            ax.set_title('%s std vs error for %s and %s split' % (title_name,
                                                                  model_type,
                                                                  splitter))
            data = results_df[results_df.dset_key == dset_name]
            std_col = [col for col in results_df.columns if 'std' in col][0]
            actual_col = [col for col in results_df.columns if 'actual' in col][0]
            g = sns.scatterplot(x="error", y='actual', hue='pred', palette="YlOrRd", data=data, ax=ax)
            ax.set(xlabel='Prediction error', ylabel='Observed value')
            pdf.savefig(fig)
        except Exception as e:
            print(e)
            continue
    
    pdf.close()
    
    pdf_path = '%s/%s_predicted_actual_std.pdf' % (plt_dir, col_name)
    pdf = PdfPages(pdf_path)
    for i, cat in enumerate(cats):
        fig = plt.figure(figsize=(10, 13))
        ax = fig.add_subplot(111)
        title_name = cat.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace('HEK_IonWorks_Electrophys_2Hz_',
                                                                               '').rstrip('.csv').replace('_', ' ')
        ax.set_title('%s Predicted vs Actual for %s with %s split' % (title_name,
                                                                      model_type,
                                                                      splitter))
        data = results_df[results_df.dset_key == cat]
        std_col = [col for col in results_df.columns if 'std' in col][0]
        actual_col = [col for col in results_df.columns if 'actual' in col][0]
        predicted_col = [col for col in results_df.columns if 'pred' in col][0]
        
        g = sns.scatterplot(x=actual_col, y=predicted_col, hue=std_col, palette="YlOrRd", data=data, ax=ax,
                            legend='brief')
        g.legend().set_title('actual')
        # replace labels
        for t in g.legend().texts: t.set_text('std' if t.get_text() == 'std' else round(float(t.get_text()), 3))
        ax.set(xlabel='Actual', ylabel='Predicted')
        pdf.savefig(fig)
    pdf.close()
    
    pdf_path = '%s/%s_uncertainties_error_bars.pdf' % (plt_dir, col_name)
    pdf = PdfPages(pdf_path)
    for i, cat in enumerate(cats):
        fig = plt.figure(figsize=(10, 13))
        ax = fig.add_subplot(111)
        df_sub = results_df[results_df['dset_key'] == cat]
        title_name = cat.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace('HEK_IonWorks_Electrophys_2Hz_',
                                                                               '').rstrip('.csv').replace('_', ' ')
        ax.scatter(df_sub['actual'], df_sub['pred'], c=df_sub['std'], cmap='gray', marker='.')
        ax.set_title('%s Predicted vs Actual for %s with %s split' % (title_name,
                                                                      model_type,
                                                                      splitter))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        eb = ax.errorbar(df_sub['actual'], df_sub['pred'], yerr=df_sub['std'], fmt='.')
        # a, (b, c), (d,) = eb.lines
        # d.set_color(col)
        pdf.savefig(fig)
    pdf.close()


# ------------------------------------------------------------------------------------------------------------------------
'''
def plot_uncertainties_binned(results_df, col_name='pilot'):
    """
    Get results for models in the given collection. For each training dataset, separate the
    results by model type and featurizer, and make a scatter plot of <subset> set r2_score vs
    model type/featurizer combo. Then make plots of r2 vs model parameters relevant for each
    model type.
    """
    res_dir = '/ds/projdata/gsk_data/model_analysis/'
    plt_dir = '%s/Plots' % res_dir
    os.makedirs(plt_dir, exist_ok=True)
    res_file = '%s_uncertainties.csv' % (col_name)
    res_path = os.path.join(res_dir, res_file)
    results_df = pd.read_csv(res_path, index_col=False)
    cats = results_df['dset_key'].unique()
    cats.sort()
    sns.set(font_scale=1.5)
    model_type = results_df.model_type.values[0]
    splitter = results_df.splitter.values[0]
    pdf_path = '%s/%s_uncertainty_plots.pdf' % (plt_dir, col_name)
    pdf = PdfPages(pdf_path)
    for dset_name in cats:
        print(dset_name)
        for model_type in ['NN', 'RF']:
            for splitter in ['scaffold','random']:
                #data = results_df[(results_df.dset_key == dset_name) & (results_df.model_type == model_type) & (results_df.splitter == splitter)]
                data = results_df[results_df.dset_key == dset_name]
                try:
                    fig = plt.figure(figsize=(10,13))
                    ax = fig.add_subplot(111)
                    title_name = dset_name.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace('HEK_IonWorks_Electrophys_2Hz_', '').rstrip('.csv').replace('_', ' ')
                    ax.set_title('%s std vs error' % (title_name))
                    std_col = [col for col in results_df.columns if 'std' in col][0]
                    actual_col = [col for col in results_df.columns if 'actual' in col][0]
                    g = sns.distplot(data, columns=[x="error", y=std_col, bins=5, hue=actual_col, palette="YlOrRd", data=data, ax=ax, legend='brief')
                    ax.set(xlabel='Binned prediction error', ylabel='Uncertainty of prediction')
                    pdf.savefig(fig)
                except Exception as e:
                    print(e)
                    continue

    pdf.close()


    pdf_path = '%s/%s_predicted_actual_std.pdf' % (plt_dir, col_name)
    pdf = PdfPages(pdf_path)
    for i,cat in enumerate(cats):
        fig = plt.figure(figsize=(10,13))
        ax = fig.add_subplot(111)
        title_name = cat.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace('HEK_IonWorks_Electrophys_2Hz_', '').rstrip('.csv').replace('_', ' ')
        ax.set_title('%s Predicted vs Actual for %s with %s split' % (title_name,
                                                              model_type,
                                                              splitter))
        data = results_df[results_df.dset_key == cat]
        std_col = [col for col in results_df.columns if 'std' in col][0]
        actual_col = [col for col in results_df.columns if 'actual' in col][0]
        predicted_col = [col for col in results_df.columns if 'pred' in col][0]

        g = sns.scatterplot(x=actual_col, y=predicted_col, hue=std_col, palette="YlOrRd", data=data, ax=ax, legend='brief')
        g.legend().set_title('actual')
        # replace labels
        for t in g.legend().texts: t.set_text('std' if t.get_text() == 'std' else round(float(t.get_text()), 3))
        ax.set(xlabel='Actual', ylabel='Predicted')
        pdf.savefig(fig)
        pdf.savefig(fig)
    pdf.close()

    pdf_path = '%s/%s_uncertainties_error_bars.pdf' % (plt_dir, col_name)
    pdf = PdfPages(pdf_path)
    for i,cat in enumerate(cats):
        fig = plt.figure(figsize=(10,13))
        ax = fig.add_subplot(111)
        df_sub = results_df[results_df['dset_key'] == cat]
        title_name = cat.split('/')[-1].replace('dskey_ATOM_GSK_', '').replace('HEK_IonWorks_Electrophys_2Hz_', '').rstrip('.csv').replace('_', ' ')
        ax.scatter(df_sub['actual'], df_sub['pred'], c=df_sub['std'], cmap='gray', marker='.')
        ax.set_title('%s Predicted vs Actual for %s with %s split' % (title_name,
                                                                              model_type,
                                                                              splitter))
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        eb = ax.errorbar(df_sub['actual'], df_sub['pred'], yerr=df_sub['std'], fmt='.')
        #a, (b, c), (d,) = eb.lines
        #d.set_color(col)
        pdf.savefig(fig)
    pdf.close()
    '''
    
#------------------------------------------------------------------------------------------------------------------------
def plot_timings(timings_file, out_dir):
    timings_df = pd.read_csv(timings_file)
    timings_df = timings_df.drop(timings_df['run_time'].idxmax())
    timings_df = timings_df.drop(timings_df['run_time'].idxmax())
    timings_df['layer_info'] = timings_df['layer_sizes'] + ' ' + timings_df['dropouts']
    def get_params(row):
        if row['model_type'] == 'RF' or row['featurizer'] == 'graphconv':
            return 0
        ls = [row['num_samples']]
        #ls = [1]
        if type(row['layer_sizes']) != str:
            return 0
        ls.extend(row['layer_sizes'].strip('[').strip(']').split(','))
        ls.append(1)
        ls = [int(l) for l in ls]
        return int(sum([(x+1)*y for x,y in list(zip(ls[:-1], ls[1:]))]))
    timings_df['num_params'] = timings_df.apply(get_params, axis=1)
    timings_df['log_num_params'] = timings_df['num_params'].apply(np.log)
    timings_df['model_feat'] = timings_df["featurizer"] + '_' + timings_df["model_type"]
    timings_df['run_time/cpds'] = timings_df['run_time'].map(float)/timings_df['num_samples'].map(float)

    sns_plot =sns.scatterplot(x='num_samples', y='run_time', data=timings_df, hue='model_feat', legend='brief')
    sns_plot.set_xlabel('Number of Compounds')
    sns_plot.set_ylabel('Runtime')
    handles, labels = sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles=handles[1:], labels=labels[1:])
    sns_plot.figure.savefig(os.path.join(out_dir, 'runtime_num_compounds.pdf'))

    sns_plot = sns.catplot(x="model_type", y="run_time", data=timings_df, height=8, hue='num_samples', legend='brief')
    sns_plot.set_xlabels('Model Type')
    sns_plot.set_ylabels('Runtime')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_model_type.pdf'))

    sns_plot = sns.catplot(x="featurizer", y="run_time", data=timings_df, height=8, hue='num_samples', legend='brief')
    sns_plot.set_xlabels('Featurizer + Model Type')
    sns_plot.set_ylabels('Runtime')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_feat.pdf'))

    sns_plot = sns.catplot(x="model_feat", y="run_time", data=timings_df, height=15, hue='num_samples', legend='brief')
    sns_plot.set_xlabels('Featurizer + Model Type')
    sns_plot.set_ylabels('Runtime')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_model_feat.pdf'))

    sns_plot = sns.catplot(x="layer_sizes", y="run_time", data=timings_df, height=15, hue='featurizer', legend='brief')
    sns_plot.set_xlabels('Layer architecture')
    sns_plot.set_ylabels('Runtime')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_layer_size.pdf'))

    sns_plot = sns.catplot(x="dropouts", y="run_time", data=timings_df, height=12, hue='featurizer', legend='brief')
    sns_plot.set_xlabels('Dropout probabilities')
    sns_plot.set_ylabels('Runtime')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_dropouts.pdf'))


    sns_plot = sns.catplot(x="layer_info", y="run_time", data=timings_df, height=15,hue='featurizer', legend='brief')
    sns_plot.set_xticklabels(rotation=60)
    sns_plot.set_xlabels('Layer architecture + dropout probabilities')
    sns_plot.set_ylabels('Runtime')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_layer_dropouts.pdf'))

    sns_plot =sns.scatterplot(x='log_num_params', y='run_time', data=timings_df[timings_df['num_params'] != np.NaN], hue='model_feat', legend='brief')
    sns_plot.set_xlabel('log(number of parameters)')
    sns_plot.set_ylabel('Runtime')
    handles, labels = sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles=handles[1:3], labels=labels[1:3])
    sns_plot.figure.savefig(os.path.join(out_dir, 'runtime_num_params.pdf'))
    
    sns_plot = sns.catplot(x="model_type", y="run_time/cpds", data=timings_df, height=8, hue='num_samples', legend='brief')
    sns_plot.set_xlabels('Model Type')
    sns_plot.set_ylabels('Runtime/Number of Compounds')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_cpds_model_type.pdf'))

    sns_plot = sns.catplot(x="featurizer", y="run_time/cpds", data=timings_df, height=8, legend='brief')
    sns_plot.set_xlabels('Featurizer')
    sns_plot.set_ylabels('Runtime/Number of Compounds')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_cpds_feat.pdf'))

    sns_plot = sns.catplot(x="model_feat", y="run_time/cpds", data=timings_df, height=15, legend='brief')
    sns_plot.set_xlabels('Featurizer + Model Type')
    sns_plot.set_ylabels('Runtime/Number of Compounds')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_cpds_model_feat.pdf'))

    sns_plot = sns.catplot(x="layer_sizes", y="run_time/cpds", data=timings_df, height=15, hue='model_feat', legend='brief')
    sns_plot.set_xlabels('Layer architecture')
    sns_plot.set_ylabels('Runtime/Number of Compounds')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_cpds_layer_size.pdf'))

    sns_plot = sns.catplot(x="dropouts", y="run_time/cpds", data=timings_df, height=12, hue='model_feat', legend='brief')
    sns_plot.set_xlabels('Dropout probabilities')
    sns_plot.set_ylabels('Runtime/Number of Compounds')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_cpds_dropouts.pdf'))

    sns_plot = sns.catplot(x="layer_info", y="run_time/cpds", data=timings_df, height=15,hue='model_feat', legend='brief')
    sns_plot.set_xticklabels(rotation=60)
    sns_plot.set_xlabels('Layer architecture + dropout probabilities')
    sns_plot.set_ylabels('Runtime/Number of Compounds')
    sns_plot.savefig(os.path.join(out_dir, 'runtime_cpds_layer_dropouts.pdf'))
    
    sns_plot =sns.scatterplot(x='log_num_params', y='run_time/cpds', data=timings_df[timings_df['num_params'] != np.NaN], hue='model_feat', legend='brief')
    sns_plot.set_xlabel('log(number of parameters)')
    sns_plot.set_ylabel('Runtime/Number of Compounds')
    handles, labels = sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles=handles[1:3], labels=labels[1:3])
    sns_plot.figure.savefig(os.path.join(out_dir, 'runtime_cpds_num_params.pdf'))
    
