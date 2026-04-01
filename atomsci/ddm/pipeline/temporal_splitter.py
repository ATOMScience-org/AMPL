"""Code to split a DeepChem dataset by assigning compounds produced before a cutoff date to the training and validation subsets,
and compounds from after the cutoff date to the test subset. This is typically used to assess training performance
under a simulated drug discovery scenario, in which models are trained on early lead compounds and used to predict properties
of compounds designed later.

Requires that the date associated with each compound be specified when constructing the splitter.

Although this class and its methods are public, you will typically not call them directly. Instead, they are invoked by
setting `splitter` to 'temporal' and setting appropriate values for the `cutoff_date`, `date_col` and `base_splitter`
parameters when you train a model.
"""

from deepchem.splits.splitters import Splitter, RandomSplitter, ScaffoldSplitter
import numpy as np
import tempfile

from atomsci.ddm.pipeline.ave_splitter import AVEMinSplitter

import logging
logging.basicConfig(format='%(asctime)-15s %(message)s')
# Set up logging
log = logging.getLogger('ATOM')



#*******************************************************************************************************************************************
class TemporalSplitter(Splitter):
    """
    Class for splitting a DeepChem dataset so that training and validation set compounds are associated with dates before a cutoff
    and test compounds have dates after the cutoff.

    Attributes:
        cutoff_date (np.datetime64 or str): Date at which to split compounds between training/validation and test sets.
        If this isn't a datetime64 object, function will attempt to convert it to one.

        date_col (str): Column where compound dates are stored in dataset attributes table.

        base_splitter (str): Type of splitter to use for partitioning training and validation compounds.

        metric (str): Name of metric to use with ave_min base splitter, if specified.

        verbose (bool): Whether to print verbose diagnostic messages.
    """

    def __init__(self, cutoff_date, date_col, base_splitter, metric=None, verbose=True):
        """
        Create a temporal splitter.

        """

        self.cutoff_date = np.datetime64(cutoff_date)
        self.date_col = date_col
        self.metric = metric
        self.verbose = verbose
        self.base_splitter_type = base_splitter
        if base_splitter == 'random':
            self.base_splitter = RandomSplitter()
        elif base_splitter == 'scaffold':
            self.base_splitter = ScaffoldSplitter()
        elif base_splitter == 'ave_min':
            self.base_splitter = AVEMinSplitter(metric=metric)

    def split(self, dataset, attr_df, frac_train=0.8, frac_valid=0.2, frac_test=0.0, seed=None, log_every_n=None):
        """
        Split the dataset into training, validation and test sets. Assigns compounds with dates after self.cutoff_date
        to the test set.  Splits the remaining compounds into training and validation sets using self.base_splitter
        with parameters frac_train and frac_valid. Note that frac_test is ignored, since the test split is based
        on dates only.

        Args:
            dataset (deepchem.Dataset): Dataset to be split.

            attr_df (pd.DataFrame): Table of compound attributes from original training set data frame. Must include
                column of dates, with label self.date_col.

            frac_train (float): Fraction of non-test compounds to put in 'train' subset.

            frac_valid (float): Fraction of non-test compounds to put in 'valid' subset.

            frac_test (float): Ignored, included only for compatibility with DeepChem Splitter API. Test set assignments
                are based on date values only.

            seed (int): Ignored, included only for compatibility with DeepChem Splitter API.

            log_every_n (int): Ignored, included only for compatibility with DeepChem Splitter API.

        Returns:
            tuple: Lists of indices for train, valid and test sets.

        """
        if self.date_col not in attr_df.columns.values:
            raise ValueError("date_col missing from dataset attributes")
        cmpd_dates = attr_df[self.date_col].values
        test_ind = np.where(cmpd_dates > self.cutoff_date)[0]

        train_valid_ind = sorted(set(range(len(cmpd_dates))) - set(test_ind))
        train_valid_frac = frac_train + frac_valid
        tv_dataset = dataset.select(train_valid_ind)
        train_ind, valid_ind, _ = self.base_splitter.split(tv_dataset, frac_train=frac_train/train_valid_frac, 
                                                           frac_valid=frac_valid/train_valid_frac, frac_test=0.0)
        log.debug("Temporal split yields %d/%d/%d train/valid/test compounds" % (len(train_ind), len(valid_ind), len(test_ind)))
        return train_ind, valid_ind, test_ind

    def train_valid_test_split(self,
                               dataset,
                               train_dir=None,
                               valid_dir=None,
                               test_dir=None,
                               frac_train=.8,
                               frac_valid=.2,
                               frac_test=np.nan,
                               seed=None,
                               log_every_n=None,
                               attr_df=None):
        """
        Splits dataset into training, validation and test sets.
        Overrides base deepchem.Splitter method to allow passing attr_df.

        Args:
            dataset (deepchem.Dataset): Dataset to be split.

            attr_df (pd.DataFrame): Table of compound attributes from original training set data frame. Must include
                column of dates, with label self.date_col.

            frac_train (float): Fraction of non-test compounds to put in 'train' subset.

            frac_valid (float): Fraction of non-test compounds to put in 'valid' subset.

            frac_test (float): Ignored, included only for compatibility with DeepChem Splitter API. Test set assignments
                are based on date values only.

            train_dir (None): Ignored, included only for compatibility with DeepChem Splitter API.

            valid_dir (None): Ignored, included only for compatibility with DeepChem Splitter API.

            test_dir (None): Ignored, included only for compatibility with DeepChem Splitter API.

            seed (None): Ignored, included only for compatibility with DeepChem Splitter API.

            log_every_n (None): Ignored, included only for compatibility with DeepChem Splitter API.

        Returns:
            tuple: Deepchem.Dataset objects for the training, validation and test subsets.

        """
        if attr_df is None:
            raise ValueError("TemporalSplitter.train_valid_test_split requires attr_df argument")

        train_inds, valid_inds, test_inds = self.split(
            dataset, attr_df, frac_train=frac_train,
            frac_valid=frac_valid, frac_test=frac_test)
        train_dataset = dataset.select(train_inds, tempfile.mkdtemp())
        if frac_valid != 0:
            valid_dataset = dataset.select(valid_inds, tempfile.mkdtemp())
        else:
            valid_dataset = None
        test_dataset = dataset.select(test_inds, tempfile.mkdtemp())

        return train_dataset, valid_dataset, test_dataset
