"""Functions to generate multi-plot displays to assess split quality: response value distributions, test-vs-train Tanimoto distance distributions, etc.
"""

import os
import numpy as np
import pandas as pd
from argparse import Namespace
from atomsci.ddm.pipeline.model_pipeline import ModelPipeline
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline import perf_plots as pp
from atomsci.ddm.utils import split_response_dist_plots as srdp
from atomsci.ddm.utils import compare_splits_plots as csp

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------------------------------------------------------------
def plot_split_diagnostics(params_or_pl, axes=None, num_rows=None, num_cols=1, min_tvt_dist=0.3, plot_size=7):
    """Generate several plots showing various aspects of split quality. These include: The value distributions for each response
    column in the different split subsets; the nearest-training-set-neighbor Tanimoto distance distributions for the validation
    and test sets; the actual split subset proportions; and (for multitask scaffold splits) the progression of fitness term values
    over generations of the genetic algorithm.
    
    Args:
        params_or_pl (argparse.Namespace or dict or ModelPipeline): Structure containing dataset and split parameters, or
        the ModelPipeline object used to perform a split. If a ModelPipeline is passed as this argument, the parameters
        are extracted from it and an additional plot will be generated for multitaskscaffold splits showing the progression
        of fitness term values over generations.  The following parameters are required, if not set to default values:
        
        | - dataset_key
        | - split_uuid
        | - split_strategy
        | - splitter
        | - split_valid_frac
        | - split_test_frac
        | - num_folds
        | - smiles_col
        | - response_cols

        axes (matplotlib.Axes): Axes to draw plots in. If provided, must contain enough entries to display all the plots requested
        (2 for each response column + 3 + 1 for a multitask scaffold split when a ModelPipeline is passed to params_or_pl). If not
        provided, a figure and Axes of the required length will be created.

        num_rows (int): Number of rows for the Axes layout; ignored if an existing set of Axes are passed in the `axes` argument.
        num_cols (int): Number of columns for the Axes layout; ignored if an existing set of Axes are passed in the `axes` argument.
        plot_size (float): Height of plots; ignored if `axes` is provided
    Returns:
        None
    """

    # Save current matplotlib color cycle and switch to 'colorblind' palette
    _ = sns.color_palette()
    sns.set_palette('colorblind')

    if isinstance(params_or_pl, dict):
        params = parse.wrapper(params_or_pl)
    elif isinstance(params_or_pl, Namespace):
        params = params_or_pl
    elif isinstance(params_or_pl, ModelPipeline):
        params = params_or_pl.params
        splitter = params_or_pl.data.splitting.splitter
        params.fitness_terms = splitter.fitness_terms
        params.split_uuid = params_or_pl.data.split_uuid
        
    else:
        raise ValueError("params_or_pl must be dict, Namespace or ModelPipeline")
    
    # Figure out how many plots we're going to generate and how to lay them out
    nresp = len(params.response_cols)
    nfitness = int((params.splitter == 'multitaskscaffold') and (isinstance(params_or_pl, ModelPipeline)))
    nplots = 2*nresp + 3 + nfitness
    if axes is not None:
        axes = axes.flatten()
        if len(axes) < nplots:
            raise ValueError(f"axes argument needs {nplots} axis pairs for requested plots; only provides {len(axes)}")
    else:
        if num_rows is None:
            num_rows = (nplots + 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*plot_size, num_rows*plot_size), layout='constrained')
        axes = axes.flatten()
    plot_num = 0

    # Draw split response distribution plots
    srdp.plot_split_subset_response_distrs(params, plot_size=plot_size, axes=axes)

    # Draw NN Tanimoto distance distribution plots with titles showing fraction of compounds below min_tvt_dist
    plot_num += nresp
    dset_path = params.dataset_key
    split_path = f"{os.path.splitext(dset_path)[0]}_{params.split_strategy}_{params.splitter}_{params.split_uuid}.csv"
    dset_df = pd.read_csv(dset_path, dtype={params.id_col: str})
    split_df = pd.read_csv(split_path, dtype={'cmpd_id': str})
    split_stats = csp.SplitStats(dset_df, split_df, smiles_col=params.smiles_col, id_col=params.id_col,
                                 response_cols=params.response_cols)
    vvtr_dists = split_stats.dists_tvv
    tvtr_dists = split_stats.dists_tvt
    vfrac = sum(vvtr_dists <= min_tvt_dist)/len(vvtr_dists)
    tfrac = sum(tvtr_dists <= min_tvt_dist)/len(tvtr_dists)
    ax = axes[plot_num]
    ax = split_stats.dist_hist_train_v_valid_plot(ax=ax)
    ax.set_title(f"Valid vs train Tanimoto NN distance distribution\nFraction <= {min_tvt_dist} = {vfrac:.3f}")

    plot_num += 1
    ax = axes[plot_num]
    ax = split_stats.dist_hist_train_v_test_plot(ax=ax)
    ax.set_title(f"Test vs train Tanimoto NN distance distribution\nFraction <= {min_tvt_dist} = {tfrac:.3f}")

    # Draw plots of actual and requested split proportions and counts for each response column after excluding missing values
    plot_num += 1
    plot_split_fractions(params, axes[plot_num:])
    plot_num += nresp

    # Plot the evolution over generations of each term in the fitness function optimized by the multitask scaffold splitter.
    if nfitness > 0:
        plot_fitness_terms(params, axes[plot_num:])

def plot_fitness_terms(params, axes):
    """Plot the evolution over generations of each term in the fitness function optimized by the multitask scaffold splitter.

    Args:
        params (argparse.Namespace or dict): Structure containing dataset and split parameters. The parameters must include
        the key `fitness_terms`, which is created by the multitask splitter; usually this is only available when `params`
        is derived from the ModelPipeline object used to split the dataset.

        axes (numpy.array of matplotlib.Axes): Axes to draw plots in.
    """
    if isinstance(params, dict):
        params = parse.wrapper(params)
    if isinstance(axes, np.ndarray):
        ax = axes[0]
    else:
        ax = axes
    fitness_df = pd.DataFrame(params.fitness_terms)
    fitness_df['generation'] = list(range(len(fitness_df)))
    fit_long_df = fitness_df.melt(id_vars='generation', value_vars=list(params.fitness_terms.keys()), var_name='Fitness term', value_name='Score')
    ax = sns.lineplot(data=fit_long_df, x='generation', y='Score', hue='Fitness term', ax=ax)
    ax.set_title('Unweighted fitness scores vs generation')
        

def plot_split_fractions(params, axes):
    """Draw plots of actual and requested split proportions and counts for each response column after excluding missing values

    Args:
        params (argparse.Namespace or dict): Structure containing dataset and split parameters.
        The following parameters are required, if not set to default values:
        
        | - dataset_key
        | - split_uuid
        | - split_strategy
        | - splitter
        | - split_valid_frac
        | - split_test_frac
        | - num_folds
        | - smiles_col
        | - response_cols

        axes (matplotlib.Axes): Axes to draw plots in. Must contain at least as many entries as response columns in the dataset.
    """
    split_dset_df, _ = srdp.get_split_labeled_dataset(params)
    axes = axes.flatten()
    req_color = pp.test_col
    actual_color = pp.train_col
    for icol, resp_col in enumerate(params.response_cols):
        ss_df = split_dset_df[split_dset_df[resp_col].notna()]
        ss_counts = ss_df.subset.value_counts()
        total_count = sum(ss_counts)
        ss_fracs = ss_counts / total_count
        actual_df = pd.DataFrame(dict(counts=ss_counts, fracs=ss_fracs)).reindex(['train', 'valid', 'test'])
        req_fracs = dict(train=1-params.split_valid_frac-params.split_test_frac, valid=params.split_valid_frac, test=params.split_test_frac)
        req_df = pd.DataFrame(dict(fracs=req_fracs)).reindex(['train', 'valid', 'test'])
        req_df['counts'] = np.round(total_count * req_df.fracs).astype(int)

        bar_width = 0.35

        # Positions of the bars on the x-axis
        r1 = np.arange(len(actual_df))
        r2 = r1 + bar_width

        # Plotting the proportions
        ax1 = axes[icol]
        bars_requested = ax1.bar(r1, req_df.fracs.values, color=req_color, width=bar_width, label='Requested', alpha=0.6)
        bars_actual = ax1.bar(r2, actual_df.fracs.values, color=actual_color, width=bar_width, label='Actual', alpha=0.6)
        max_count = max(max(actual_df.counts.values), max(req_df.counts.values))

        # Left Y-axis (proportions)
        ax1.set_ylabel('Proportions')
        #ax1.tick_params(axis='y')

        # Create the second Y-axis (counts)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Counts')
        #ax2.tick_params(axis='y', labelcolor='k')
        ax2.set_ylim(0, max_count*1.1)

        # Adding counts on top of the bars
        for bar, count in zip(bars_requested, req_df.counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height, f'{count}', ha='center', va='bottom', color=actual_color)

        for bar, count in zip(bars_actual, actual_df.counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height, f'{count}', ha='center', va='bottom', color=req_color)

        # X-axis labels
        plt.xticks([r + bar_width / 2 for r in range(len(actual_df))], actual_df.index.values)

        # Title and legend
        if icol == 0:
            plt.title(f"Requested and Actual Subset Proportions and Counts\n{resp_col}")
            ax1.legend(loc='upper right')
        else:
            plt.title(f"\n{resp_col}")
