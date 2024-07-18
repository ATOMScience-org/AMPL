"""Utility functions used for AMPL dataset curation and creation."""

""" TOC:
aggregate_assay_data(assay_df, value_col='VALUE_NUM', output_value_col=None,
                         label_actives=True,
                         active_thresh=None,
                         id_col='CMPD_NUMBER', smiles_col='rdkit_smiles', relation_col='VALUE_FLAG', date_col=None)
replicate_rmsd(dset_df, smiles_col='base_rdkit_smiles', value_col='PIC50', relation_col='relation')
mle_censored_mean(cmpd_df, std_est, value_col='PIC50', relation_col='relation')
get_three_level_class(value, red_thresh, yellow_thresh)
get_binary_class(value, thresh=4.0)
set_group_permissions(path, system='AD', owner='GSK')
filter_in_by_column_values (column, values, data)
filter_out_by_column_values (column, values, data)
filter_out_comments (values, values_cs, data) ...delete rows that contain comments listed (can specify 'case sensitive' if needed)
get_rdkit_smiles_parent (data)...................creates a new column with the rdkit smiles parent (salts stripped off)
average_and_remove_duplicates (column, tolerance, list_bad_duplicates, data)
summarize_data(column, num_bins, title, units, filepath, data)..............prints mix/max/avg/histogram
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles

feather_supported = True
try:
    import pyarrow.feather as feather
except (ImportError, AttributeError, ModuleNotFoundError):
    feather_supported = False


import matplotlib.pyplot as plt

# ******************************************************************************************************************************************
def set_group_permissions(path, system='AD', owner='GSK'):
    """Sets file and group permissions to standard values for a dataset containing proprietary
    data owned by 'owner'. Later we may add a 'public' option, or groups for data from other pharma companies.

    Args:
        path (string): File path
        
        system (string): Computing environment from which group ownerships will be derived; currently, either 'LC' for LC
        filesystems or 'AD' for LLNL systems where owners and groups are managed by Active Directory.
        
        owner (string): Who the data belongs to, either 'public' or the name of a company (e.g. 'GSK') associated with a
        restricted access group.

    Returns:
        None
    """

    # Currently, if we're not on an LC machine, we're on an AD-controlled system. This could change.
    if system != 'LC':
        system = 'AD'
    owner_group_map = dict(GSK = {'LC' : 'gskcraa', 'AD' : 'gskusers-ad'},
                           public = {'LC' : 'atom', 'AD' : 'atom'} )
    group = owner_group_map[owner][system]
    shutil.chown(path, group=group)
    os.chmod(path, 0o770)

# ******************************************************************************************************************************************
def replicate_rmsd(dset_df, smiles_col='base_rdkit_smiles', value_col='PIC50', relation_col='relation', default_val=1.0):
    """Compute RMS deviation of all replicate uncensored measurements from means

    Compute RMS deviation of all replicate uncensored measurements in dset_df from their means. Measurements are treated
    as replicates if they correspond to the same SMILES string, and are considered censored if the relation
    column contains > or <. The resulting value is meant to be used as an estimate of measurement error for all compounds
    in the dataset.

    Args:
        dset_df (DataFrame): DataFrame containing uncensored measurements and SMILES strings.

        smiles_col (str): Name of the column that contains SMILES strings.

        value_col (str): Name of the column that contains target values.

        relation_col (str): The input DataFrame column containing relational operators (<, >, etc.).

        default_val (float): The value to return if there are no compounds with replicate measurements.

    Returns:
        float: returns root mean squared deviation of all replicate uncensored measurements

    """
    dset_df = dset_df[~(dset_df[relation_col].isin(['<', '>']))]
    uniq_smiles, uniq_counts = np.unique(dset_df[smiles_col].values, return_counts=True)
    smiles_with_reps = uniq_smiles[uniq_counts > 1]
    if len(smiles_with_reps) > 0:
        uniq_devs = []
        for smiles in smiles_with_reps:
            values = dset_df[dset_df[smiles_col] == smiles][value_col].values
            uniq_devs.extend(values - values.mean())
        uniq_devs = np.array(uniq_devs)
        rmsd = np.sqrt(np.mean(uniq_devs ** 2))
    else:
        rmsd = default_val
    return rmsd

# ******************************************************************************************************************************************
def mle_censored_mean(cmpd_df, std_est, value_col='PIC50', relation_col='relation'):
    """Computes maximum likelihood estimate of the true mean value for a single replicated compound.

    Compute a maximum likelihood estimate of the true mean value underlying the distribution of replicate assay measurements for a
    single compound. The data may be a mix of censored and uncensored measurements, as indicated by the 'relation' column in the input
    DataFrame cmpd_df. std_est is an estimate for the standard deviation of the distribution, which is assumed to be Gaussian;
    we typically compute a common estimate for the whole dataset using replicate_rmsd().

    Args:
        cmpd_df (DataFrame): DataFrame containing measurements and SMILES strings.

        std_est (float): An estimate for the standard deviation of the distribution.

        smiles_col (str): Name of the column that contains SMILES strings.

        value_col (str): Name of the column that contains target values.

        relation_col (str): The input DataFrame column containing relational operators (<, >, etc.).

    Returns:
        float: maximum likelihood estimate of the true mean for a replicated compound
        str: Relation, '' not censored, '>' right censored, '<' left censored
    """
    left_censored = np.array(cmpd_df[relation_col].values == '<', dtype=bool)
    right_censored = np.array(cmpd_df[relation_col].values == '>' , dtype=bool)
    not_censored = ~(left_censored | right_censored)
    n_left_cens = sum(left_censored)
    n_right_cens = sum(right_censored)
    nreps = cmpd_df.shape[0]
    values = cmpd_df[value_col].values
    nan = float('nan')

    relation = ''
    # If all the replicate values are left- or right-censored, return the smallest or largest reported (threshold) value accordingly.
    if n_left_cens == nreps:
        mle_value = min(values)
        relation = '<'
    elif n_right_cens == nreps:
        mle_value = max(values)
        relation = '>'
    elif n_left_cens + n_right_cens == 0:
        # If no values are censored, the MLE is the actual mean.
        mle_value = np.mean(values)
    else:
        # Some, but not all observations are censored.
        # First, define the negative log likelihood function
        def loglik(mu):
            ll = -sum(norm.logpdf(values[not_censored], loc=mu, scale=std_est))
            if n_left_cens > 0:
                ll -= sum(norm.logcdf(values[left_censored], loc=mu, scale=std_est))
            if n_right_cens > 0:
                ll -= sum(norm.logsf(values[right_censored], loc=mu, scale=std_est))
            return ll

        # Then minimize it
        opt_res = minimize_scalar(loglik, method='brent')
        if not opt_res.success:
            if 'message' in opt_res.keys():
                print('Likelihood maximization failed, message is: "%s"' % opt_res.message)
            else:
                print('Likelihood maximization failed')
            mle_value = nan
        else:
            mle_value = opt_res.x
    return mle_value, relation


# ******************************************************************************************************************************************
def aggregate_assay_data(assay_df, value_col='VALUE_NUM', output_value_col=None,
                         label_actives=True,
                         active_thresh=None,
                         id_col='CMPD_NUMBER', smiles_col='rdkit_smiles', relation_col='VALUE_FLAG', date_col=None, verbose=False):
    """Aggregates replicated values in assay data

    Map RDKit SMILES strings in assay_df to base structures, then compute an MLE estimate of the mean value over replicate measurements
    for the same SMILES strings, taking censoring into account. Generate an aggregated result table with one value for each unique base
    SMILES string, to be used in an ML-ready dataset.

    Args:
        assay_df (DataFrame): The input DataFrame to be processed.

        value_col (str): The column in the DataFrame containing assay values to be averaged.

        output_value_col (str): Optional; the column name to use in the output DataFrame for the averaged data.

        label_actives (bool): If True, generate an additional column 'active' indicating whether the mean value is above a threshold specified by active_thresh.

        active_thresh (float): The threshold to be used for labeling compounds as active or inactive.
            If active_thresh is None (the default), the threshold used is the minimum reported value across all records
            with left-censored values (i.e., those with '<' in the relation column.

        id_col (str): The input DataFrame column containing compound IDs.

        smiles_col (str): The input DataFrame column containing SMILES strings.

        relation_col (str): The input DataFrame column containing relational operators (<, >, etc.).

        date_col (str): The input DataFrame column containing dates when the assay data was uploaded. If not None, the code will assign the earliest
            date among replicates to the aggregate data record.

    Returns:
        A DataFrame containing averaged assay values, with one value per compound.
    """

    assay_df = assay_df.fillna({relation_col: '', smiles_col: ''})
    # Filter out rows where SMILES is missing
    n_missing_smiles = np.array([len(smiles) == 0 for smiles in assay_df[smiles_col].values]).sum()
    if verbose:
        print("%d entries in input table are missing SMILES strings" % n_missing_smiles)
    has_smiles = np.array([len(smiles) > 0 for smiles in assay_df[smiles_col].values])
    assay_df = assay_df[has_smiles].copy()

    # Estimate the measurement error across replicates for this assay
    std_est = replicate_rmsd(assay_df, smiles_col=smiles_col, value_col=value_col, relation_col=relation_col)

    # Map SMILES strings to base structure SMILES strings, then map these to indices into the list of
    # unique base structures
    orig_smiles_strs = assay_df[smiles_col].values
    norig = len(set(orig_smiles_strs))
    smiles_strs = [base_smiles_from_smiles(smiles, True) for smiles in orig_smiles_strs]
    assay_df['base_rdkit_smiles'] = smiles_strs
    uniq_smiles_strs = sorted(set(smiles_strs))
    nuniq = len(uniq_smiles_strs)
    if verbose:
        print("%d unique SMILES strings are reduced to %d unique base SMILES strings" % (norig, nuniq))
    smiles_map = dict([(smiles,i) for i, smiles in enumerate(uniq_smiles_strs)])
    smiles_indices = np.array([smiles_map.get(smiles, nuniq) for smiles in smiles_strs])

    assay_vals = assay_df[value_col].values
    value_flags = assay_df[relation_col].values

    # Compute a maximum likelihood estimate of the mean assay value for each compound, averaging over replicates
    # and factoring in censoring. Report the censoring/relation/value_flag only if the flags are consistent across
    # all replicates.  # Exclude compounds that couldn't be mapped to SMILES strings.

    cmpd_ids = assay_df[id_col].values
    reported_cmpd_ids = ['']*nuniq
    reported_value_flags = ['']*nuniq
    if date_col is not None:
        reported_dates = ['']*nuniq
    reported_assay_val = np.zeros(nuniq, dtype=float)
    for i in range(nuniq):
        cmpd_ind = np.where(smiles_indices == i)[0]
        cmpd_df = assay_df.iloc[cmpd_ind]
        reported_assay_val[i], reported_value_flags[i] = mle_censored_mean(cmpd_df, std_est, value_col=value_col,
                                                                           relation_col=relation_col)
        # When multiple compound IDs map to the same base SMILES string, use the lexicographically smallest one.
        reported_cmpd_ids[i] = sorted(set(cmpd_ids[cmpd_ind]))[0]

        # If a date column is specified, use the earliest one among replicates
        if date_col is not None:
            # np.datetime64 doesn't seem to understand the date format in GSK's crit res tables
            #earliest_date = sorted([np.datetime64(d) for d in cmpd_df[date_col].values])[0]
            earliest_date = sorted(pd.to_datetime(cmpd_df[date_col], infer_datetime_format=True).values)[0]
            reported_dates[i] = np.datetime_as_string(earliest_date)

    if output_value_col is None:
        output_value_col = value_col
    agg_df = pd.DataFrame({
                   'compound_id' : reported_cmpd_ids,
                   'base_rdkit_smiles' : uniq_smiles_strs,
                   'relation' : reported_value_flags,
                   output_value_col : reported_assay_val})

    if date_col is not None:
        agg_df[date_col] = reported_dates

    # Label each compound as active or not, based on the reported relation and values relative to a common threshold
    if label_actives:
        inactive_df = agg_df[agg_df.relation == '<']
        if inactive_df.shape[0] > 0 and active_thresh is None:
            active_thresh = np.min(inactive_df[output_value_col].values)
        if active_thresh is not None:
            is_active = ((agg_df.relation != '<') & (agg_df[output_value_col].values > active_thresh))
            agg_df['active'] = [int(a) for a in is_active]
        else:
            agg_df['active'] = 1

    return agg_df

# ******************************************************************************************************************************************
def freq_table(dset_df, column, min_freq=1):
    """Generate a DataFrame tabluating the repeat requencies of unique values.

    Generate a DataFrame tabulating the repeat frequencies of each unique value in 'column'.
    Restrict it to values occurring at least min_freq times.

    Args:
        dset_df (DataFrame): An input DataFrame

        column (str): The name of one column in DataFrame

        min_freq (int): Restrict unique count to at least min_freq times.

    Returns:
        DataFrame: Dataframe containing two columns: the column passed in as the 'column' argument
            and the column 'Count'. The 'Count' column contains the number of occurances for each
            value in the 'column' argument.
    """
    nmissing = sum(dset_df[column].isna())
    filt_df = dset_df[dset_df[column].notna()]
    uniq_vals, counts = np.unique(filt_df[column].values, return_counts=True)
    uniq_vals = uniq_vals.tolist() + [np.nan]
    counts = counts.tolist() + [nmissing]
    uniq_df = pd.DataFrame({column: uniq_vals, 'Count': counts}).sort_values(by='Count', ascending=False)
    uniq_df = uniq_df[uniq_df.Count >= min_freq]
    return uniq_df

# ******************************************************************************************************************************************
def labeled_freq_table(dset_df, columns, min_freq=1):
    """Generate a frequency table in which additional columns are included.

    Generate a frequency table in which additional columns are included. The first column in 'columns'
    is assumed to be a unique ID; there should be a many-to-1 mapping from the ID to each of the additional
    columns.

    Args:
        dset_df (DataFrame): The input DataFrame.

        columns (list(str)): A list of columns to include in the output frequency table.
            The first column in 'columns' is assumed to be a unique ID; there should be
            a many-to-1 mapping from the ID to each of the additional columns.

        min_freq (int): Restrict unique count to at least min_freq times.

    Returns:
        DataFrame: A DataFrame containing a frequency table.

    Raises:
        Exception: If the DataFrame violates the rule: there should be a many-to-1
            mapping from the ID to each of the additional columns.

    """
    id_col = columns[0]
    freq_df = freq_table(dset_df, id_col, min_freq=min_freq)
    uniq_ids = freq_df[id_col].values
    addl_cols = columns[1:]
    addl_vals = {colname: [] for colname in addl_cols}
    uniq_df = dset_df.drop_duplicates(subset=columns)
    for uniq_id in uniq_ids:
        subset_df = uniq_df[uniq_df[id_col] == uniq_id]
        if subset_df.shape[0] > 1:
            raise Exception("Additional columns should be unique for ID %s" % uniq_id)
        for colname in addl_cols:
            addl_vals[colname].append(subset_df[colname].values[0])
    for colname in addl_cols:
        freq_df[colname] = addl_vals[colname]
    return freq_df

# ******************************************************************************************************************************************
# The functions below are from Claire Weber's data_utils module.

# ******************************************************************************************************************************************
def filter_in_out_by_column_values(column, values, data, in_out):
    """Include rows only for given values in specified column.

    Given a DataFrame, column, and an iterable, Series, DataFrame, or dict, of values,
    return a DataFrame with rows containing value in values or all rows
    that do not containe a value in values.

    Args:
        column (str): Name of a column in data.

        values (iterable): An iterable, Series, DataFrame, or dict of values
            contained in data[column].

        data (DataFrame): A DataFrame.

        in_out (str): If set to 'in', will filter in rows that contain a value
            in values. If set to anything else, this function will filter out
            rows that contian a value in values.

    Returns:
        DataFrame: DataFrame containing filtered rows.
    """

    if in_out == 'in':
        data = data.loc[data[column].isin (values)]
    else:
        data = data.loc[~data[column].isin (values)]

    return data


# ******************************************************************************************************************************************
def filter_in_by_column_values (column, values, data):
    """Include rows only for given values in specified column.

    Filters in all rows in data if row[column] in values.

    Args:
        column (str): Name of a column in data.

        values (iterable): An iterable, Series, DataFrame, or dict of values
            contained in data[column].

        data (DataFrame): A DataFrame.

    Returns:
        DataFrame: DataFrame containing filtered rows.
    """
    return filter_in_out_by_column_values (column, values, data, 'in')


# ******************************************************************************************************************************************
def filter_out_by_column_values (column, values, data):
    """Exclude rows only for given values in specified column.

    Filters out all rows in data if row[column] in values.

    Args:
        column (str): Name of a column in data.

        values (iterable): An iterable, Series, DataFrame, or dict of values
            contained in data[column].

        data (DataFrame): A DataFrame.

    Returns:
        DataFrame: DataFrame containing filtered rows.
    """
    return filter_in_out_by_column_values (column, values, data, 'out')


# ******************************************************************************************************************************************
def filter_out_comments (values, values_cs, data):
    """Remove rows that contain the text listed

    Removes any rows where data['COMMENTS'] contains the words in
    values or values_cs. Used for removing results that indicate
    bad data in the comments.

    Args:
        values (str): list of values that are not case sensitive

        values_cs (str): list of values that are case sensitive

        data (DataFrame): DataFrame containing a column named 'COMMENTS'

    Returns:
        DataFrame: Returns a DataFrame with the remaining rows
    """

    column = 'COMMENTS'
    data['Remove'] = np.where (data[column].str.contains ('|'.join (values), case=False), 1, 0)
    data['Remove'] = np.where (data[column].str.contains ('|'.join (values_cs), case=True), 1, data['Remove'])
    data['Remove'] = np.where (data[column].str.contains ('nan', case=False), 0, data['Remove'])
    data['Remove'] = np.where (data[column] == ' ', 0, data['Remove'])
    data_removed = data[data.Remove == 1]
    data = data[data.Remove != 1]
    data_removed = data_removed['COMMENTS']
    #print(data_removed)
    del data['Remove']

    # Results
    #print ("")
    #print('Remove results with comments indicating bad data')
    #print("Dataframe size", data.shape[:])
    #comments = pd.DataFrame(data['COMMENTS'].unique())
    #comments = comments.sort_values(comments.columns[0])
    #print (comments) # For the purpose of reviewing comments remaining

    return data


# ******************************************************************************************************************************************
# DEPRECATED: This is extremely inefficient and inflexible. Probably this is only used in some legacy curation notebooks.
def get_rdkit_smiles_parent (data):
    """Strip the salts off the rdkit SMILES strings

    First, loops through data and determines the base/parent smiles string for each row.
    Appends the base smiles string to a new row in a list.
    Then adds the list as a new column, 'rdkit_smiles_parent', in 'data'.
    Basically calls base_smiles_from_smiles for each smile in the column 'rdkit_smiles'

    Args:
        data (DataFrame): A DataFrame with a column named 'rdkit_smiles'.

    Returns:
        DataFrame with column 'rdkit_smiles_parent' with salts stripped
    """
    print ("")
    print ("Adding SMILES column 'rdkit_smiles_parent' with salts stripped...(may take a while)", flush=True)

    i_max = data.shape[0]
    rdkit_smiles_parent = []
    for i in range (i_max):
        smile = data['rdkit_smiles'].iloc[i]
        if type (smile) is float:
            split = ''
        else:
            split = base_smiles_from_smiles (smile)

        rdkit_smiles_parent.append (split)

    #  2. Add base smiles string (stripped smiles) to dataset
    data['rdkit_smiles_parent'] = rdkit_smiles_parent

    return data


# ---------------------------------------------------------------------------------------------------------------------------------
def remove_outlier_replicates(df, response_col='pIC50', id_col='compound_id', max_diff_from_median=1.0):
    """Examine groups of replicate measurements for compounds identified by compound ID and compute median response
    for each group. Eliminate measurements that differ by more than a given value from the median; note that
    in some groups this will result in all replicates being deleted. This function should be used together with
    `aggregate_assay_data` instead of `average_and_remove_duplicates` to reduce data to a single value per compound.

    Args:
        df (DataFrame): Table of compounds and response data

        response_col (str): Column containing response values

        id_col (str): Column that uniquely identifies compounds, and therefore measurements to be treated as replicates.

        max_diff_from_median (float): Maximum absolute difference from median value allowed for retained replicates.

    Returns:
        result_df (DataFrame): Filtered data frame with outlier replicates removed.

    """

    fr_df = freq_table(df, id_col, min_freq=2)
    rep_ids = fr_df[id_col].values.tolist()
    has_rep_df = df[df[id_col].isin(rep_ids)]
    no_rep_df = df[~df[id_col].isin(rep_ids)]
    gby = has_rep_df.groupby(id_col)
    def filter_outliers(g_df):
        med = np.median(g_df[response_col].values)
        keep = ( np.abs( g_df[response_col].values - med ) <= max_diff_from_median)
        return g_df[keep]
    filt_df = gby.apply(filter_outliers)
    n_removed = len(has_rep_df) - len(filt_df)
    if n_removed > 0:
        print(f"Removed {n_removed} {response_col} replicate measurements that were > {max_diff_from_median} from median")
    result_df = pd.concat([filt_df, no_rep_df], ignore_index=True)
    return result_df

# ******************************************************************************************************************************************
def average_and_remove_duplicates (column, tolerance, list_bad_duplicates,
        data, max_stdev = 100000, compound_id='CMPD_NUMBER', 
        rm_duplicate_only=False, smiles_col='rdkit_smiles_parent'):
    """This while loop loops through until no 'bad duplicates' are left.

    This function removes duplicates based on max_stdev and tolerance. If the
    value in data[column] falls too far from the mean based on tolerance and
    max_stdev then that entry is removed. This is repeated until all bad
    entries are removed

    Args:
        column (str): column with the value of interest

        tolerance (float): acceptable % difference between value and average
            ie.: if "[(value - mean)/mean*100]>tolerance" then remove data row

        list_bad_duplicates (str): 'Yes' to list the bad duplicates

        data (DataFrame): input DataFrame

        max_stdev (float): maximum standard deviation threshold

        compound_id (str): column containing compound ids

        rm_duplicate_only (bool): only remove bad duplicates, don't average good ones, the resulting table can be fed into aggregate assay data to further process.
            note: The mean is recalculated on each loop through to make sure it isn't skewed by the 'bad duplicate' values

        smiles_col (str): column containing base rdkit smiles strings

    Returns:
        DataFrame: Returns remaining rows after all bad duplicates have been removed.

    """

    list_bad_duplicates = list_bad_duplicates
    i = 0
    bad_duplicates = 1
    removed = []
    removed = pd.DataFrame(removed) 

    while i < 1 or bad_duplicates !=0 and not data.empty :
        #a. reset table if needed
        if i > 0:
            del data['VALUE_NUM_mean']
            del data['VALUE_NUM_std']
            del data['Perc_Var']
            del data['Remove_BadDuplicate']

        # 1. Calculate mean of duplicates
        unique_smiles = data.groupby(smiles_col)
        VALUE_NUM_mean = unique_smiles[column].mean()
        VALUE_NUM_std = unique_smiles[column].std()
        temporary_data = pd.concat([VALUE_NUM_mean,VALUE_NUM_std],axis=1)
        temporary_data.columns = ["VALUE_NUM_mean","VALUE_NUM_std"]
        temporary_data.reset_index(level=0, inplace=True)

        # 2. Add columns for mean back to main data file
        data = pd.merge(data, temporary_data, how='left', on=smiles_col)

        # 3. Add column for percent variance (value - mean)/value*100
        data['Perc_Var'] = (abs(data[column] - data['VALUE_NUM_mean'])/data['VALUE_NUM_mean'])*100

        # 4. Make removal recommendations
        data['Remove_BadDuplicate'] = np.where((data['Perc_Var']>tolerance),1,0)
        data['Remove_BadDuplicate'] = np.where((data['VALUE_NUM_std']>max_stdev),1, data.Remove_BadDuplicate.values)

        bad_duplicates = data['Remove_BadDuplicate'].max()  # 0 = no bad duplicates, 1 = bad duplicates

        to_remove = data.loc[data['Remove_BadDuplicate'] == 1]

        # 5. Remove bad duplicates
        data = data[data.Remove_BadDuplicate != 1]

        removed = pd.concat([removed, to_remove])
        i = i+1

        # 6. If bad duplicates were removed, loop back to step 'a.' to reset table & re-calc. If no bad duplicates, exit 'while loop'.


    #print results
    print("Bad duplicates removed from dataset")
    print("Dataframe size", data.shape[:])

    if list_bad_duplicates == 'Yes':
        print("List of 'bad' duplicates removed")
        col = [compound_id, column, 'VALUE_NUM_mean', 'Perc_Var', 'VALUE_NUM_std']
        removed = removed.sort_values(compound_id)
        print( removed[col])

    # retain only instance of each unique rdkit_smiles_parent
    if not rm_duplicate_only:
        data = data.drop_duplicates(subset=smiles_col)
        print("")
        print("Dataset de-duplicated")
        print("Dataframe size", data.shape[:])
        print("New column created with averaged values: ", 'VALUE_NUM_mean')

    return data


# ******************************************************************************************************************************************
def summarize_data(column, num_bins, title, units, filepath, data, log_column = 'No'):
    """Summarizes the in data[column]

    Summarizes the data by printing mean, stdev, max, and min of the data. Creates
    plots of the binned values in data[column]. If log_column != 'No' this also
    creates plots that compares normal and log distributions of the data.

    Args:
        column (str): Column of interest.

        num_bins (int): Number of bins in the histogram.

        title (str): Title of the histogram.

        units (str): Units for values in 'column'.

        filepath (str): This file path gets printed to the console.

        data (DataFrame): Input DataFrame.

        log_column (str): Defaults to 'No'. Any other value will generate
            a plot comparing normal and log distributions.

    Returns:
        None

    """

    dataset_mean = data[column].mean()
    dataset_max = data[column].max()
    dataset_min = data[column].min()
    dataset_std = data[column].std()

    print('Post-processing dataset')
    if filepath != "" :
       print('file source: ', filepath)
    print("")
    print("Total Number of results =", data.shape[0])
    print("dataset mean =", dataset_mean, units)
    print("dataset stdev =", dataset_std, units)
    print("dataset max =", dataset_max, units)
    print("dataset min =", dataset_min, units)
    print("")

    if 'classification' in data.columns:
        print('___Data Counts by Classification___( 0 = low)')
        print(data.groupby('classification').classification.count())

    plt.hist(data[column], num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


    if log_column != 'No':

        logify_data = data[data[column] > 0]
        removed = len(data)-len(logify_data)
        print('___Comparison of normal vs log distributions____')
        if removed > 0:
            print('''***NOTE: To logify, values equal to or less than 0 removed. Data removed from plot only - not from dataset.
              ''', removed, "results removed.")

        fig, ax = plt.subplots(1,2,figsize=(14, 5))

        plt.subplot(121)
        plot1=plt.hist(data[column], edgecolor='k',linewidth=1.0,color='blue')
        plt.title(column)
        plt.xlabel('Value')
        plt.ylabel('Count')

        plt.subplot(122)
        plot2=plt.hist(logify_data[log_column], edgecolor='k',linewidth=1.0,color='green')
        plt.title(log_column)
        plt.xlabel('Value')
        plt.ylabel('Count')

        plt.show()

# ******************************************************************************************************************************************
def create_new_rows_for_extra_results ( extra_result_col, value_col, data):
    """Moves results from an extra column to an existing column

    Returns a new DataFrame with values from 'extra_result_col' appended to the
    end of 'value_col'. NaN values in 'extra_result_col' are dropped. 'Extra_result_col'
    is dropped from the resulting DataFrame

    Args:
        extra_result_col (str): A column in 'data'.
        
        value_col (str): A column in 'data'.
        
        data (DataFrame):

    Returns:
        DataFrame
    """
    addrows = data
    addrows = addrows.dropna(subset=[extra_result_col])
    addrows = addrows.drop(columns = value_col)
    addrows.rename(columns={extra_result_col: value_col}, inplace=True)
    data = pd.concat([data,addrows])

    return data

# ******************************************************************************************************************************************
# Generalized function to assign class labels based on thresholds on a continous value column.

def add_classification_column(thresholds, value_column, label_column, data, right_inclusive=True):
    """Add a classification column to a DataFrame.

    Add a classification column 'label_column' to DataFrame 'data' based on values in 'value_column',
    according to a sequence of thresholds. The number of classes is one plus the number of thresholds.

    Args:
        thresholds (float or sequence of floats): Thresholds to use to assign class labels. Label i will
            be assigned to values such that thresholds[i-1] < value <= thresholds[i] (if right_inclusive is True)
            or thresholds[i-1] <= value < thresholds[i] (otherwise).

        value_column (str): Name of the column from which class labels are derived.

        label_column (str): Name of the new column to be created for class labels.

        data (DataFrame): DataFrame holding all data.

        right_inclusive (bool): Whether the thresholding intervals are closed on the right or on the left.
            Set this False to get the same behavior as add_binary_tertiary_classification. The default behavior
            is preferred for the common case where the classification is based on a left-censoring threshold.

    Returns:
        DataFrame: DataFrame updated to include class label column.

    """

    try:
        thresholds = sorted(thresholds)
    except TypeError:
        # raised if thresholds is scalar
        thresholds = [thresholds]

    nclasses = len(thresholds)
    values = data[value_column].values
    labels = np.zeros(len(values))
    for i, thresh in enumerate(thresholds):
        if right_inclusive:
            labels[values > thresh] = i+1
        else:
            labels[values >= thresh] = i+1
    labels[np.isnan(values)] = np.nan
    data[label_column] = labels
    return data


# ******************************************************************************************************************************************
def xc50topxc50_for_nm(x) :
   """Convert XC50 values measured in nanomolars to -log10 (PX50)

   Args :
        x (float): input XC50 value measured in nanomolars

   Returns :
        float: -log10 value of x
   """
   return -np.log10((x/1000000000.0))
