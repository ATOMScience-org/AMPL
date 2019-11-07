"""
Code to split a DeepChem dataset to select training and validation compounds from before a cutoff date
and test compounds from after the cutoff. Requires that the date associated with each compound be 
specified when constructing the splitter.
"""

from deepchem.splits.splitters import Splitter, RandomSplitter, ScaffoldSplitter
import os
import numpy as np
import pandas as pd
import random
from time import asctime
import tempfile
import pdb

from atomsci.ddm.utils import datastore_functions as dsf
from atomsci.ddm.pipeline import featurization as feat
from atomsci.ddm.pipeline import model_datasets as md
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
    """

    def __init__(self, cutoff_date, date_col, base_splitter, metric=None, verbose=True):
        """
        Create a temporal splitter.

        Params:
            cutoff_date (np.datetime64 or str): Date at which to split compounds between training/validation and test sets.
            If this isn't a datetime64 object, function will attempt to convert it to one.

            date_col (str): Column where compound dates are stored in dataset attributes table.

            base_splitter (str): Type of splitter to use for partitioning training and validation compounds.

            metric (str): Name of metric to use with ave_min base splitter, if specified.

            verbose (bool): Whether to print verbose diagnostic messages.
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

    def split(self, dataset, attr_df, frac_train=0.8, frac_valid=0.2, frac_test=0.0, log_every_n=None):
        """
        Split the dataset into training, validation and test sets. Use a temporal split to select the test set.  Then split the
        training and validation sets using self.base_splitter. Note that frac_test is ignored, since the test split is based
        on dates only.
        """
        if not (self.date_col in attr_df.columns.values):
            raise ValueError("date_col missing from dataset attributes")
        cmpd_dates = attr_df[self.date_col].values
        test_ind = np.where(cmpd_dates > self.cutoff_date)[0]
        #pdb.set_trace()
        test_dataset = dataset.select(test_ind)
        train_valid_ind = sorted(set(range(len(cmpd_dates))) - set(test_ind))
        train_valid_frac = frac_train + frac_valid
        tv_dataset = dataset.select(train_valid_ind)
        train_ind, valid_ind, _ = self.base_splitter.split(tv_dataset, frac_train=frac_train/train_valid_frac, 
                                                           frac_valid=frac_valid/train_valid_frac, frac_test=0.0)
        log.debug("Temporal split yields %d/%d/%d train/valid/test compounds" % (len(train_ind), len(valid_ind), len(test_ind)))
        return train_ind, valid_ind, test_ind

    def train_valid_test_split(self,
                               dataset,
                               attr_df,
                               frac_train=.8,
                               frac_valid=.2,
                               frac_test=np.nan,
                               verbose=True):
        """
        Overrides base deepchem Splitter method to allow passing attr_df.
        Splits dataset into train/validation/test sets.

        Returns a tuple of deepchem Dataset objects.
        """
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
