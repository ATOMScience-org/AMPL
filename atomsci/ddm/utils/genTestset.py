# genTestset.py
#
# Generate a test set for testing models built  
# on DTC, Excape and ChEMBL data sets 
#
# Written by Ya Ju Fan (fan4@llnl.gov) 
# May 20, 2020
#
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np

import atomsci.ddm.pipeline.chem_diversity as cd
import atomsci.ddm.utils.curate_data as curate_data


#=====================================
# Union of three lists 
def Union(lst1, lst2, lst3):
    final_list = list(set().union(lst1, lst2, lst3))
    return final_list
#-----------------------------
def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 
#------------------------------

# Input dataframes of three data sources:
# dtc_df, ex_df, ch_df
# Returning the dataframes:
# testset_dtc_df, testset_ex_df, testset_ch_df, trainset_dtc_df,trainset_ex_df,trainset_ch_df 
def generateTestset(dtc_df,ex_df,ch_df,test_fraction = 0.15):

    n_dtc = dtc_df.shape[0]
    n_ex = ex_df.shape[0]
    n_ch = ch_df.shape[0]

    sumOfthree = dtc_df.shape[0] + ex_df.shape[0] + ch_df.shape[0]
    print("Number of total samples: ",sumOfthree)

    dtc_s = dtc_df['base_rdkit_smiles'].tolist()
    ex_s = ex_df['base_rdkit_smiles'].tolist()
    ch_s = ch_df['base_rdkit_smiles'].tolist()

    #============================
    # Union data sets in order
    #============================
    union_df = pd.concat([dtc_df['base_rdkit_smiles'],ex_df['base_rdkit_smiles'],ch_df['base_rdkit_smiles']],ignore_index=True).drop_duplicates().reset_index(drop=True)
    unionOrder_s = union_df.tolist()
    print("Number of unique rdkit smiles in all three data sets: ", union_df.shape[0])
    print("Number of rdkit smiles in the union set: ",len(unionOrder_s))
    n_union = union_df.shape[0]

    union_df = union_df.to_frame()

    # Compute nearest distances to each of the data sources 
    feat_type='ECFP'
    dist_metric='tanimoto'
    calc_type='nearest'
    #calc_type='all'
    unionOrder_dtc_nndist=cd.calc_dist_smiles(feat_type,dist_metric,unionOrder_s,dtc_s,calc_type)
    unionOrder_ex_nndist=cd.calc_dist_smiles(feat_type,dist_metric,unionOrder_s,ex_s,calc_type)
    unionOrder_ch_nndist=cd.calc_dist_smiles(feat_type,dist_metric,unionOrder_s,ch_s,calc_type)
    # Sum of the nearest distances to the other two data sources 
    unionOrder_all_nndist = unionOrder_dtc_nndist + unionOrder_ex_nndist + unionOrder_ch_nndist

    # Add sum of distances as a column to the dataframe 
    union_df['sumDist'] = unionOrder_all_nndist
    print(unionOrder_all_nndist.shape)
    print(union_df.shape)
    print(len(unionOrder_s))

    #----------------------------
    # Randomly select a subset
    #----------------------------   
    fraction = 0.4
    subset_tmp_df = union_df.sample(frac = fraction)
    print("Number of randomely selected samples: ",subset_tmp_df.shape[0])


    subset_tmp_s = subset_tmp_df['base_rdkit_smiles'].tolist()
    # Find intersect with dtc, excape and chembl
    subset_dtc = intersection(subset_tmp_s, dtc_s)
    subset_ex = intersection(subset_tmp_s, ex_s)
    subset_ch = intersection(subset_tmp_s,ch_s)


    #=====================================================
    # Compute distances to training 

    union_df['temp_split'] = 'train'
    union_df.loc[union_df['base_rdkit_smiles'].isin(subset_tmp_s),'temp_split'] = 'test'


    union_train_s = union_df.loc[union_df['temp_split']=='train','base_rdkit_smiles']
    union_test_s = union_df.loc[union_df['temp_split']=='test','base_rdkit_smiles']

    feat_type='ECFP'
    dist_metric='tanimoto'
    calc_type='nearest'
    testDist2train=cd.calc_dist_smiles(feat_type,dist_metric,union_test_s,union_train_s,calc_type)
    print('distance from test set to training set, length: ', testDist2train.shape)


    subset_tmp_df['dist2train'] = testDist2train
    subset_tmp_df = subset_tmp_df.sort_values(by=['dist2train'], ascending=False)

    #============================================================
    # Sample equal proportions starting from the top distances
    # Force assign groups
    subset_tmp_df['source'] = 'dtc'
    subset_tmp_df.loc[subset_tmp_df['base_rdkit_smiles'].isin(ex_s),'source'] = 'excape'
    subset_tmp_df.loc[subset_tmp_df['base_rdkit_smiles'].isin(ch_s),'source'] = 'chembl' 
    subset_tmp_df.loc[subset_tmp_df['base_rdkit_smiles'].isin(dtc_s),'source'] = 'dtc'

    subset_dtc_df = subset_tmp_df.loc[subset_tmp_df['source']=='dtc']
    subset_ex_df = subset_tmp_df.loc[subset_tmp_df['source']=='excape']
    subset_ch_df = subset_tmp_df.loc[subset_tmp_df['source']=='chembl']

    subset_dtc_df.sort_values(['dist2train','sumDist'], inplace=True,ascending=False)
    subset_ex_df.sort_values(['dist2train','sumDist'], inplace=True,ascending=False)
    subset_ch_df.sort_values(['dist2train','sumDist'], inplace=True,ascending=False)


    #================
    # Fraction = 0.15
    # test_fraction = 0.15
    nSample_dtc = round(n_union*test_fraction*n_dtc/sumOfthree)
    nSample_ex = round(n_union*test_fraction*n_ex/sumOfthree)
    nSample_ch = round(n_union*test_fraction*n_ch/sumOfthree)

    subset_sel_dtc_df = subset_dtc_df.head(nSample_dtc)
    subset_sel_ex_df = subset_ex_df.head(nSample_ex)
    subset_sel_ch_df = subset_ch_df.head(nSample_ch)

    subset_sel_dtc = subset_sel_dtc_df['base_rdkit_smiles']
    subset_sel_ex = subset_sel_ex_df['base_rdkit_smiles']
    subset_sel_ch = subset_sel_ch_df['base_rdkit_smiles']

    dtc_set = set(subset_sel_dtc)
    ex_set = set(subset_sel_ex)
    ch_set = set(subset_sel_ch)

    # Find the true source groups
    # Union three selected subsets
    subset_sel_s = Union(dtc_set,ex_set,ch_set)
    subset_sel_df = subset_tmp_df.loc[subset_tmp_df['base_rdkit_smiles'].isin(subset_sel_s)]

    subset_sel_dtc_df = subset_sel_df.loc[subset_sel_df['base_rdkit_smiles'].isin(subset_dtc)]
    subset_sel_ex_df = subset_sel_df.loc[subset_sel_df['base_rdkit_smiles'].isin(subset_ex)]
    subset_sel_ch_df = subset_sel_df.loc[subset_sel_df['base_rdkit_smiles'].isin(subset_ch)]


    # Gather original dataframe features 
    testset_dtc_df = dtc_df.loc[dtc_df['base_rdkit_smiles'].isin(subset_sel_s)]
    trainset_dtc_df = dtc_df.loc[~dtc_df['base_rdkit_smiles'].isin(subset_sel_s)]
    testset_ex_df = ex_df.loc[ex_df['base_rdkit_smiles'].isin(subset_sel_s)]
    trainset_ex_df = ex_df.loc[~ex_df['base_rdkit_smiles'].isin(subset_sel_s)]
    testset_ch_df = ch_df.loc[ch_df['base_rdkit_smiles'].isin(subset_sel_s)]
    trainset_ch_df = ch_df.loc[~ch_df['base_rdkit_smiles'].isin(subset_sel_s)]
    #testset_union_df = union_df.loc[union_df['base_rdkit_smiles'].isin(subset_sel_s)]
    #trainset_union_df = union_df.loc[~union_df['base_rdkit_smiles'].isin(subset_sel_s)]

    return testset_dtc_df,testset_ex_df, testset_ch_df, trainset_dtc_df,trainset_ex_df,trainset_ch_df




#-------------------------------
# Aggregate base rdkit smiles
#------------------------------- 
def aggregate_basesmiles(assay_df, value_col='pIC50', output_value_col=None,
                         label_actives=True,
                         active_thresh=None,
                         id_col='compound_id', smiles_col='base_rdkit_smiles', relation_col='VALUE_FLAG', date_col=None):
    """Extract from aggregate_assay_data() without the need to compute base_smiles_from_smiles().
    Compute an MLE estimate of the mean value over rep
    licate measurements
    for the same SMILES strings, taking censoring into account. Generate an aggregated result table with one value f
    or each unique base
    SMILES string, to be used in an ML-ready dataset.

    Args:
        assay_df: The input data frame to be processed.
        value_col: The column in the data frame containing assay values
            to be averaged.
        output_value_col: Optional; the column name to use in the output
            data frame for the averaged data.
        label_actives: If True, generate an additional column 'active'
            indicating whether the mean value is above a threshold
            specified by active_thresh.
        active_thresh: The threshold to be used for labeling compounds
            as active or inactive.
        id_col: The input data frame column containing compound IDs.
        smiles_col: The input data frame column containing SMILES
            strings.
        relation_col: The input data frame column containing relational
            operators (<, >, etc.).
        date_col: The input data frame column containing dates when the
            assay data was uploaded. If not None, the code will assign
            the earliest
    If active_thresh is None (the default), the threshold used is the minimum reported value across all records
    with left-censored values (i.e., those with '<' in the relation column.
    date among replicates to the aggregate data record.

    Returns:
        A data frame containing averaged assay values, with one value
        per compound.
    """

    assay_df = assay_df.fillna({relation_col: '', smiles_col: ''})
    # Filter out rows where SMILES is missing
    n_missing_smiles = np.array([len(smiles) == 0 for smiles in assay_df[smiles_col].values]).sum()
    print("%d entries in input table are missing SMILES strings" % n_missing_smiles)
    has_smiles = np.array([len(smiles) > 0 for smiles in assay_df[smiles_col].values])
    assay_df = assay_df[has_smiles].copy()

    # Estimate the measurement error across replicates for this assay
    std_est = curate_data.replicate_rmsd(assay_df, smiles_col=smiles_col, value_col=value_col, relation_col=relation_col)

    # --  Modified by Yaru --
    # Map SMILES strings to base structure SMILES strings, then map these to indices into the list of
    # unique base structures
    #orig_smiles_strs = assay_df[smiles_col].values
    #norig = len(set(orig_smiles_strs))
    #smiles_strs = [base_smiles_from_smiles(smiles, True) for smiles in orig_smiles_strs]
    smiles_strs = assay_df[smiles_col].values
    norig = len(set(smiles_strs))
    # -- End  --
    assay_df['base_rdkit_smiles'] = smiles_strs
    uniq_smiles_strs = list(set(smiles_strs))
    nuniq = len(uniq_smiles_strs)
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
        reported_assay_val[i], reported_value_flags[i] = curate_data.mle_censored_mean(cmpd_df, std_est, value_col=value_col,
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



#-----------------------------------------------------------------------------------
# Generate union data sets for training and testing, using data curation functions
#-----------------------------------------------------------------------------------
def genUnionSet(dtc_df,ex_df,ch_df):
    # Combining the test sets
    combine_df = pd.concat([dtc_df,ex_df,ch_df],ignore_index=True).drop_duplicates().reset_index(drop=True)
    print('Total number of base smiles strings in combined data: ', combine_df.shape)

    tolerance=10
    column='pIC50' #'standard_value'
    list_bad_duplicates='No'
    max_std=1

    union_df=aggregate_basesmiles(combine_df, value_col=column, output_value_col=None,
                             label_actives=True,
                             active_thresh=None,
                             id_col='compound_id', smiles_col='base_rdkit_smiles', relation_col='relation')

    union_df = union_df[~union_df.isin([np.inf]).any(1)]
    
    return union_df
 
