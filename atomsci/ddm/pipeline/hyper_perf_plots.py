"""Functions for visualizing hyperparameter performance. These functions work with
a dataframe of model performance metrics and hyperparameter specifications from
compare_models.py. For models on the tracker, use get_multitask_perf_from_tracker().
For models in the file system, use get_filesystem_perf_results().
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Create an array with the colors you want to use
colors = ["#7682A4","#A7DDD8","#373C50","#694691","#BE2369","#EB1E23","#6EC8BE","#FFC30F",]
# Set your custom color palette
pal=sns.color_palette(colors)
sns.set_palette(pal)
plt.rcParams.update({"figure.dpi": 96})

from atomsci.ddm.pipeline import parameter_parser as pp

# get all possible things to plot from parameter parser
parser=pp.get_parser()
d=vars(pp.get_parser().parse_args([]))
keywords=['AttentiveFPModel','GCNModel','GraphConvModel','MPNNModel','PytorchMPNNModel','rf_','xgb_']
plot_dict={}
for word in keywords:
    tmplist=[x for x in d.keys() if x.startswith(word)]
    if word=='rf_': word='RF'
    elif word=='xgb_':word='xgboost'
    plot_dict[word]=tmplist
plot_dict['general']=['model_type','features','splitter']#'ecfp_radius',
plot_dict['NN']=['avg_dropout','learning_rate','num_weights','num_layers','best_epoch','max_epochs']

# list score types
regselmets=[
 'r2_score',
 'mae_score',
 'rms_score',
]
classselmets = [
 'roc_auc_score',
 'prc_auc_score',
 'precision',
 'recall_score',
 'npv',
 'accuracy_score',
 'kappa',
 'matthews_cc',
 'bal_accuracy',
]


def get_score_types():
    """Helper function to show score type choices.
    """
    print("Classification metrics: ", classselmets)
    print("Regression metrics: ", regselmets)


def _prep_perf_df(df):
    """This function splits columns that contain lists into individual columns to use for plotting later.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or get_filesystem_perf_results().
        
    Returns:
        perf_track_df (pd.DataFrame): a new df with modified and extra columns.
    """
    perf_track_df=df.copy()

    if 'model_parameters_dict' in perf_track_df:
        # reset the index of perf_track_df so the dataframes merge correctly
        perf_track_df.reset_index(drop=True, inplace=True)
        exp=pd.DataFrame(perf_track_df.model_parameters_dict.tolist())
        exp['model_uuid']=perf_track_df.model_uuid
        perf_track_df=perf_track_df.merge(exp)
    
    if 'NN' in perf_track_df.model_type.unique():
        
        cols=['dummy_nodes_1','dummy_nodes_2','dummy_nodes_3']
        
        # dropouts
        tmp=perf_track_df.dropouts.astype(str).str.strip('[]').str.split(pat=',', expand=True).astype(float)
        n=len(tmp.columns)
        perf_track_df[cols[0:n]]=tmp
        perf_track_df['avg_dropout']=perf_track_df[cols[0:n]].mean(axis=1)
        
        # layer sizes
        tmp= perf_track_df.layer_sizes.astype(str).str.strip('[]').str.split(pat=',', expand=True).astype(float)
        perf_track_df[cols[0:n]]=tmp
        perf_track_df['num_layers'] = n-perf_track_df[cols[0:n]].isna().sum(axis=1)
        perf_track_df[cols[0:n]]=perf_track_df[cols[0:n]].fillna(value=1).astype(int)
        perf_track_df['num_weights']=perf_track_df[cols[0:n]].product(axis=1)
        perf_track_df.num_weights=perf_track_df.num_weights.astype(float)
        # perf_track_df=perf_track_df.drop(columns=cols[0:n])
        
        perf_track_df.loc[perf_track_df.model_type != "NN", 'layer_sizes']=np.nan
        perf_track_df.loc[perf_track_df.model_type != "NN", 'num_layers']=np.nan
        perf_track_df.loc[perf_track_df.model_type != "NN", 'num_weights']=np.nan
        perf_track_df.loc[perf_track_df.model_type != "NN", 'avg_dropout']=np.nan
    
    return perf_track_df


def plot_train_valid_test_scores(df, prediction_type='regression'):
    """This function creates line plots of performance scores based on their partitions.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or get_filesystem_perf_results().
        
        prediction_type (str): the type of model you want to visualize. Valid options are "regression" and "classification".  
    """
    sns.set_context('notebook')
    perf_track_df=df.copy().reset_index(drop=True)

    if prediction_type=='regression':
        selmets=regselmets
    elif prediction_type=='classification':
        selmets=classselmets
        
    nrows=perf_track_df.splitter.nunique()
    
    with sns.axes_style("ticks"):
        fig, ax = plt.subplots(nrows,len(selmets), figsize=(5*len(selmets),5*nrows))
        if nrows>1:
            for i, splitter in enumerate(perf_track_df.splitter.unique()):
                for j, scoretype in enumerate(selmets):
                    plot_df=perf_track_df[perf_track_df.splitter==splitter]
                    plot_df=plot_df[[f"best_train_{scoretype}",f"best_valid_{scoretype}",f"best_test_{scoretype}"]]
                    plot_df=plot_df.sort_values(f"best_valid_{scoretype}")
                    ax[i,j].plot(plot_df.T)
                    ax[i,j].set_ylim(plot_df.min().min()-.1,1.25)
                    ax[i,j].tick_params(rotation=15)
                    ax[i,j].set_title(f'{splitter} {scoretype}')
        else:
            splitter=perf_track_df.splitter.iloc[0]
            for j, scoretype in enumerate(selmets):
                plot_df=perf_track_df[perf_track_df.splitter==splitter]
                plot_df=plot_df[[f"best_train_{scoretype}",f"best_valid_{scoretype}",f"best_test_{scoretype}"]]
                plot_df=plot_df.sort_values(f"best_valid_{scoretype}")
                ax[j].plot(plot_df.T)
                ax[j].set_ylim(plot_df.min().min()-.1,1.25)
                ax[j].tick_params(rotation=15)
                ax[j].set_title(f'{splitter} {scoretype}')
            
        fig.suptitle("Model performance by partition")
        plt.tight_layout()

    
def plot_split_perf(df, prediction_type='regression', subset='valid'):
    """This function creates boxplots of performance scores based on the splitter type.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or get_filesystem_perf_results().
        
        prediction_type (str): the type of model you want to visualize. Valid options are "regression" and "classification".  
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context("notebook")    
    perf_track_df=_prep_perf_df(df).reset_index(drop=True)
    
    if prediction_type=='regression':
        selmets=regselmets
    elif prediction_type=='classification':
        selmets=classselmets
        
    plot_df=perf_track_df
    plot_df=plot_df.sort_values('features')
    with sns.axes_style("ticks"):
        fig, axes = plt.subplots(1,len(selmets), figsize=(5*len(selmets),5))
        for i, ax in enumerate(axes.flat):
            if i==len(axes.flat)-1:
                legend=True
            else:
                legend=False
            selection_metric = f'best_{subset}_{selmets[i]}'
            sns.boxplot(x="features", y=selection_metric, # x="txptr_features" x="model_type"
                        hue='splitter', palette = sns.color_palette(colors[0:plot_df.splitter.nunique()]), #showfliers=False, 
                          legend=legend,
                        data=plot_df, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel(selection_metric.replace(f'best_{subset}_',''))
            ax.set_xticks(ax.get_xticks()) # avoid warning by including this line
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor' )
            ax.set_title(selection_metric.replace(f'best_{subset}_',''))
            if legend: sns.move_legend(ax, loc=(1.01,0.5))
        plt.tight_layout()
        fig.suptitle('Effect of splitter on model performance', y=1.01)


def plot_hyper_perf(df, scoretype='r2_score', subset='valid', model_type='general'):
    """This function creates boxplots or scatter plots of performance scores based on their hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in get_score_types()
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.

        model_type (str): the type of model you want to visualize. Options include 'general' (features and splits), 'RF', 'NN', 'xgboost', and limited functionality for 'AttentiveFPModel', 'GCNModel', 'GraphConvModel', 'MPNNModel', and 'PytorchMPNNModel'.
    """

    sns.set_context('notebook')
    perf_track_df=_prep_perf_df(df).reset_index(drop=True)
    if model_type !='general':
        perf_track_df=perf_track_df[perf_track_df.model_type==model_type]
    winnertype= f'best_{subset}_{scoretype}'

    feats=plot_dict[model_type]
    ncols=3
    nrows=int(np.ceil(len(feats)/ncols))

    helix_dict={'NN':(0,0.40),'RF':(0,2.00),'xgboost':(0,2.75),'general':(60,0.2)}

    if model_type=='xgboost':
        for feat in feats:
            try: perf_track_df[feat]=perf_track_df[feat].round(3)
            except: continue
                
    fig, ax = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*4))
    ax=ax.ravel()
    for i, feat in enumerate(feats):
        try: rot,start=helix_dict[model_type]
        except: rot,start=(-0.2,0)
        if feat in perf_track_df.columns:    
            if perf_track_df[feat].nunique()>12:
                sns.scatterplot(x=feat, y=winnertype, data=perf_track_df, ax=ax[i])
                old_ticks=ax[i].get_xticks()
                old_labs=ax[i].get_xticklabels()
                ticks=[]
                labs=[]
                for tick, lab in zip(old_ticks, old_labs):
                    if tick>=0:
                        ticks.append(tick)
                        labs.append(lab)
            else:       
                sns.boxplot(x=feat,y=winnertype,hue=feat,palette=sns.cubehelix_palette(perf_track_df[feat].nunique(), rot=rot,start=start,), data=perf_track_df, ax=ax[i],legend=False)
                ticks=ax[i].get_xticks()
                labs=ax[i].get_xticklabels()
            if feat != 'num_weights':
                ax[i].set_xticks(ticks) # avoid warning by including this line
                ax[i].set_xticklabels(labs, rotation=30, ha='right', rotation_mode='anchor' )
        ax[i].set_xlabel(feat)
    fig.suptitle(f'{model_type} hyperparameter performance')
    plt.tight_layout()

### the following 3 plots are originally from Amanda M.
def plot_rf_perf(df, scoretype='r2_score',subset='valid'):
    """This function plots scatterplots of performance scores based on their RF hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in get_score_types()
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context('poster')
    perf_track_df=df.copy().reset_index(drop=True)
    plot_df=perf_track_df[perf_track_df.model_type=='RF'].copy()
    winnertype= f'best_{subset}_{scoretype}'
    
    if len(plot_df)>0:
        feat1 = 'rf_estimators'; feat2 = 'rf_max_features'; feat3 = 'rf_max_depth'
        plot_df[f'{feat1}_cut']=pd.qcut(plot_df[feat1], 5, precision=0)
        plot_df[f'{feat2}_cut']=pd.qcut(plot_df[feat2], 5, precision=0)
        hue=feat3
        plot_df = plot_df.sort_values([f'{feat1}_cut', f'{feat2}_cut',feat3], ascending=True)
        plot_df.loc[:,f'{feat1}/{feat2}'] = ['%s / %s' % (mf,est) for mf,est in zip(plot_df[f'{feat1}_cut'], plot_df[f'{feat2}_cut'])]
        with sns.axes_style("whitegrid"):
            fig,ax = plt.subplots(1,figsize=(40,15))
            if plot_df[hue].nunique()<13:
                palette=sns.cubehelix_palette(plot_df[hue].nunique())
            else:
                palette=sns.cubehelix_palette(as_cmap=True)
            sns.scatterplot(x=f'{feat1}/{feat2}', y=winnertype, hue=hue, palette=palette, data=plot_df, ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=30, ha='right')
            ax.set_title('RF model performance')
    else: print("There are no RF models in this set.")

        
def plot_nn_perf(df, scoretype='r2_score',subset='valid'):
    """This function plots scatterplots of performance scores based on their NN hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in get_score_types()
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context('poster')
    perf_track_df=_prep_perf_df(df).reset_index(drop=True)
    plot_df=perf_track_df[perf_track_df.model_type=='NN'].copy()
    winnertype= f'best_{subset}_{scoretype}'
    
    if len(plot_df)>0:
        feat1 = 'num_weights'; feat2 = 'learning_rate'; feat3 = 'avg_dropout'
        plot_df[f'{feat1}_cut']=pd.qcut(plot_df[feat1],5)
        plot_df[f'{feat2}_cut']=pd.qcut(plot_df[feat2],5)
        plot_df = plot_df.sort_values([f'{feat1}_cut', f'{feat2}_cut',feat3], ascending=True)
        bins=['{:,}'.format(int(round(x,-3))) for x in pd.qcut(plot_df[feat1],5,retbins=True)[1]]
        bins.pop(0)
        bins1=[0]
        bins1.extend(bins)
        bins2=['{:.2e}'.format(x) for x in pd.qcut(plot_df[feat2],5,retbins=True)[1]]
        for bins, feat in zip([bins1,bins2],[feat1,feat2]):
            binstrings=[]
            for i,bin in enumerate(bins):
                try:binstrings.append(f'({bin}, {bins[i+1]}]')
                except:pass
            nncmap=dict(zip(plot_df[f'{feat}_cut'].dtype.categories.tolist(),binstrings))
            plot_df[f'{feat}_cut']=plot_df[f'{feat}_cut'].map(nncmap)
        hue=feat3
        plot_df[f'{feat1}/{feat2}'] = ['%s / %s' % (mf,est) for mf,est in zip(plot_df[f'{feat1}_cut'], plot_df[f'{feat2}_cut'])]
        with sns.axes_style("whitegrid"):
            fig,ax = plt.subplots(1,figsize=(40,15))
            if plot_df[hue].nunique()<13:
                palette=sns.cubehelix_palette(plot_df[hue].nunique())
            else:
                palette=sns.cubehelix_palette(as_cmap=True)
            sns.scatterplot(x=f'{feat1}/{feat2}', y=winnertype, hue=hue,
                            palette=palette, 
                            data=plot_df, ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=30, ha='right')
            ax.set_title('NN model performance')
    else: print("There are no NN models in this set.")


def plot_xg_perf(df, scoretype='r2_score',subset='valid'):
    """This function plots scatterplots of performance scores based on their XG hyperparameters.
    
    Args:
        df (pd.DataFrame): A dataframe containing model performances from a hyperparameter search. Best practice is to use get_multitask_perf_from_tracker() or get_filesystem_perf_results().
        
        scoretype (str): the score type you want to use. Valid options can be found in get_score_types()
        
        subset (str): the subset of scores you'd like to plot from 'train', 'valid' and 'test'.
    """
    sns.set_context('poster')
    perf_track_df=df.copy().reset_index(drop=True)
    plot_df=perf_track_df[perf_track_df.model_type=='xgboost'].copy()
    winnertype= f'best_{subset}_{scoretype}'
    if len(plot_df)>0:
        feat1 = 'xgb_learning_rate'; feat2 = 'xgb_gamma'
        for feat in [feat1,feat2]:
            plot_df[feat]=plot_df[feat].round(3)
        hue=feat2
        plot_df = plot_df.sort_values([feat1, feat2])
        #plot_df[f'{feat1}/{feat2}'] = ['%s / %s' % (mf,est) for mf,est in zip(plot_df[feat1], plot_df[feat2])]
        with sns.axes_style("whitegrid"):
            fig,ax = plt.subplots(1,figsize=(40,15))
            if plot_df[hue].nunique()<13:
                palette=sns.cubehelix_palette(plot_df[hue].nunique())
            else:
                palette=sns.cubehelix_palette(as_cmap=True)
            sns.scatterplot(x=feat1, y=winnertype, hue=hue, 
                                palette=palette, 
                                data=plot_df, ax=ax)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=30, ha='right')
            ax.set_title('XGboost model performance')
    else: print('There are no XGBoost models in this set.')