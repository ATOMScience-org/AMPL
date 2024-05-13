import argparse
import pandas as pd
import os
import numpy as np

from atomsci.ddm.pipeline import chem_diversity as cd
from atomsci.ddm.pipeline import MultitaskScaffoldSplit as mss

import seaborn as sns
from matplotlib import pyplot
from rdkit import Chem
from rdkit.Chem import AllChem
import umap

class SplitStats:
    """This object manages a dataset and a given split dataframe."""
    def __init__(self, total_df, split_df, smiles_col, id_col, response_cols):
        """Calculates compound to compound Tanomoto distances between training and
        test subsets. Counts the number of samples for each subset, for each task
        and calculates the train_frac, valid_frac, and test_frac.

        Args:
            total_df (DataFrame): Pandas DataFrame.
            split_df (DataFrame): AMPL split data frame. Must contain
                'cmpd_id' and 'subset' columns.
            smiles_col (str): SMILES column in total_df.
            id_col (str): ID column in total_df.
            response_cols (str): Response columns in total_df.
        """
        self.smiles_col = smiles_col
        self.id_col = id_col
        self.response_cols = response_cols
        self.total_df = total_df
        self.split_df = split_df

        self.train_df, self.test_df, self.valid_df = split(self.total_df, self.split_df, self.id_col)

        self.total_y, self.total_w = mss.make_y_w(self.total_df, response_cols)
        self.train_y, self.train_w = mss.make_y_w(self.train_df, response_cols)
        self.test_y, self.test_w = mss.make_y_w(self.test_df, response_cols)
        self.valid_y, self.valid_w = mss.make_y_w(self.valid_df, response_cols)

        self.dists_tvt = self._get_dists(self.test_df, self.train_df)
        self.dists_tvv = self._get_dists(self.valid_df, self.train_df)

        self.train_fracs, self.valid_fracs, self.test_fracs = self._split_ratios()

    def _get_dists(self, df_a, df_b):
        """Calculate Tanimoto distances between each compound in df_a and its nearest neighbor in df_b.

        Args:
            df_a: choice of self.train_df, self.test_df, self.valid_df
            df_b: choice of self.train_df, self.test_df, self.valid_df

        Returns:
            1-D array of floats with one element per row of df_a, containing nearest neighbor
            Tanimoto distances.
        """
        return cd.calc_dist_smiles('ECFP', 'tanimoto', df_a[self.smiles_col].values, 
                    df_b[self.smiles_col].values)
    
    def _split_ratios(self):
        """Calculates the fraction of samples belonging to training, validation, and test subsets.

        Args:
            None

        Returns:
            train_fracs (array of floats), valid_fracs (array of floats), test_fracs (array of floats)
        """
        train_fracs = np.sum(self.train_w, axis=0)/np.sum(self.total_w, axis=0)
        valid_fracs = np.sum(self.valid_w, axis=0)/np.sum(self.total_w, axis=0)
        test_fracs = np.sum(self.test_w, axis=0)/np.sum(self.total_w, axis=0)
    
        return train_fracs, valid_fracs, test_fracs

    def print_stats(self):
        """Prints useful statistics to stdout"""
        print("dist tvt mean: %0.2f, median: %0.2f, std: %0.2f"%\
            (np.mean(self.dists_tvt), np.median(self.dists_tvt), np.std(self.dists_tvt)))
        print("dist tvv mean: %0.2f, median: %0.2f, std: %0.2f"%\
            (np.mean(self.dists_tvv), np.median(self.dists_tvv), np.std(self.dists_tvv)))
        print("train frac mean: %0.2f, median: %0.2f, std: %0.2f"%\
           (np.mean(self.train_fracs), np.median(self.train_fracs), np.std(self.train_fracs)))
        print("test frac mean: %0.2f, median: %0.2f, std: %0.2f"%\
            (np.mean(self.test_fracs), np.median(self.test_fracs), np.std(self.test_fracs)))
        print("valid frac mean: %0.2f, median: %0.2f, std: %0.2f"%\
            (np.mean(self.valid_fracs), np.median(self.valid_fracs), np.std(self.valid_fracs)))

    def dist_hist_train_v_test_plot(self, ax=None):
        """Plots histogram of nearest neighbor Tanimoto distances between test and training subset compounds.

        Args:
            ax (matploblib Axes): Axes object to draw plot in. If None, one will be created.

        Returns:
            ax (matploblib Axes): Axes object for plot
        """
        return self._show_dist_hist_plot(self.dists_tvt, ax=ax)

    def dist_hist_train_v_valid_plot(self, ax=None):
        """Plots histogram of nearest neighbor Tanimoto distances between valid and training subset compounds.

        Args:
            ax (matploblib Axes): Axes object to draw plot in. If None, one will be created.

        Returns:
            ax (matploblib Axes): Axes object for plot
        """
        return self._show_dist_hist_plot(self.dists_tvv, ax=ax)

    def dist_hist_plot(self, dists, title, dist_path=''):
        """Creates a histogram of pairwise Tanimoto distances between training
        and test sets

        Args:
            dist_path (str): Optional Where to save the plot. The string '_dist_hist' will be
                appended to this input
        """
        # plot compound distance histogram
        fig=pyplot.figure()
        g = self._show_dist_hist_plot(dists)
        fig.suptitle(title)        
        if len(dist_path) > 0:
            save_figure(dist_path+'_dist_hist')
        pyplot.close()

    def _show_dist_hist_plot(self, dists, ax=None):
        """Creates a histogram of pairwise Tanimoto distances between training
        and test sets

        Args:
            dists (np.ndarray): array of distances, either self.dists_tvt or self.dists_tvv

            ax (matploblib Axes): Axes object to draw plot in. If None, one will be created.

        Returns:
            ax (matploblib Axes): Axes object for plot

        """
        ax=sns.histplot(dists, kde=False, stat='probability', binrange=(0,1), ax=ax)
        ax.set_xlabel('Tanimoto distance',fontsize=13)
        ax.set_ylabel('Proportion of compounds',fontsize=13)

        return ax

    def umap_plot(self, dist_path=''):
        """Plots the first 10000 samples in Umap space using Morgan Fingerprints

        Args:
            dist_path (str): Optional Where to save the plot. The string '_umap_scatter' will be
                appended to this input
        """
        # umap of a subset
        sub_sample_df = self.split_df.loc[np.random.permutation(self.split_df.index)[:10000]]
        # add subset column to total_df
        sub_total_df = sub_sample_df[['cmpd_id', 'subset']].merge(
            self.total_df, left_on='cmpd_id', right_on=self.id_col, how='inner')
        fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024) for s in sub_total_df[self.smiles_col]]
        fp_array = np.array(fp)

        embedded = umap.UMAP().fit_transform(fp_array)
        sub_total_df['x'] = embedded[:,0]
        sub_total_df['y'] = embedded[:,1]
        pyplot.figure()
        sns.scatterplot(x='x', y='y', hue='subset', data=sub_total_df)
        if len(dist_path) > 0:
            save_figure(dist_path+'_umap_scatter')

        pyplot.close()

    def subset_frac_plot(self, dist_path=''):
        """Makes a box plot of the subset fractions

        Args:
            dist_path (str): Optional Where to save the plot. The string '_frac_box' will be
                appended to this input
        """
        dicts = []
        for f in self.train_fracs:
            dicts.append({'frac':f, 'subset':'train'})
        for f in self.test_fracs:
            dicts.append({'frac':f, 'subset':'test'})
        for f in self.valid_fracs:
            dicts.append({'frac':f, 'subset':'valid'})

        frac_df = pd.DataFrame.from_dict(dicts)

        pyplot.figure()
        g = sns.boxplot(x='subset', y='frac', data=frac_df)
        if len(dist_path) > 0:
            save_figure(dist_path+'_frac_box')

    def make_all_plots(self, dist_path=''):
        """Makes a series of diagnostic plots

        Args:
            dist_path (str): Optional Where to save the plot. The string '_frac_box' will be
                appended to this input
        """
        # histogram of compound distances between training, valid, and test subsets
        self.dist_hist_plot(self.dists_tvt, 'Train vs Test pairwise Tanimoto Distance',
            dist_path=dist_path+'_tvt')
        self.dist_hist_plot(self.dists_tvv, 'Train vs Valid pairwise Tanimoto Distance',
            dist_path=dist_path+'_tvv')

        # umap on ecfp fingerprints. visualizes clusters of training/valid/testing split
        self.umap_plot(dist_path)

        # box plot of fractions
        self.subset_frac_plot(dist_path)

def split(total_df, split_df, id_col):
    """Splits a dataset into training, test and validation sets using a given split.

    Args:
        total_df (DataFrame): A pandas dataframe.
        split_df (DataFrame): A split dataframe containing 'cmpd_id' and 'subset' columns.
        id_col (str): The ID column in total_df

    Returns:
        (DataFrame, DataFrame, DataFrame): Three dataframes for train, test, and valid
            respectively.
    """
    train_df = total_df[total_df[id_col].isin(split_df[split_df['subset']=='train']['cmpd_id'])]
    test_df = total_df[total_df[id_col].isin(split_df[split_df['subset']=='test']['cmpd_id'])]
    valid_df = total_df[total_df[id_col].isin(split_df[split_df['subset']=='valid']['cmpd_id'])]
    
    return train_df, test_df, valid_df

def save_figure(filename):
    """Saves a figure to disk. Saves both png and svg formats.

    Args:
        filename (str): The name of the figure.
    """
    pyplot.tight_layout()
    pyplot.savefig(filename+'.png')
    pyplot.savefig(filename+'.svg')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('csv', help='Source dataset csv.')
    parser.add_argument('id_col', help='ID column for source dataset')
    parser.add_argument('smiles_col', help='SMILES column for source dataset')
    parser.add_argument('split_a', help='Split A. A split csv generated by AMPL')
    parser.add_argument('split_b', help='Split B. A split csv generated by AMPL')

    parser.add_argument('output_dir', help='Output directory for plots')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    df = pd.read_csv(args.csv, dtype={args.id_col:str})

    split_a = pd.read_csv(args.split_a, dtype={'cmpd_id':str})
    ss = SplitStats(df, split_a, smiles_col=args.smiles_col, id_col=args.id_col)
    ss.make_all_plots(dist_path=os.path.join(args.output_dir, 'split_a'))


    split_b = pd.read_csv(args.split_b, dtype={'cmpd_id':str})
    ss = SplitStats(df, split_b, smiles_col=args.smiles_col, id_col=args.id_col)
    ss.make_all_plots(dist_path=os.path.join(args.output_dir, 'split_b'))
