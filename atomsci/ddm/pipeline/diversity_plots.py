"""Plotting routines for visualizing chemical diversity of datasets"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import umap
from scipy.stats.kde import gaussian_kde
from scipy.cluster.hierarchy import linkage
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from atomsci.ddm.utils import struct_utils
from atomsci.ddm.pipeline import dist_metrics as dm
from atomsci.ddm.pipeline import  chem_diversity as cd
from atomsci.ddm.utils import datastore_functions as dsf

#matplotlib.style.use('ggplot')
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize=12)

logging.basicConfig(format='%(asctime)-15s %(message)s')

ndist_max = 1000000

#------------------------------------------------------------------------------------------------------------------
def plot_dataset_dist_distr(dataset, feat_type, dist_metric, task_name, **metric_kwargs):
    """
    Generate a density plot showing the distribution of distances between dataset feature
    vectors, using the specified feature type and distance metric.

    Args:
        dataset (deepchem.Dataset): A dataset object. At minimum, it should contain a 2D numpy array 'X' of feature vectors.

        feat_type (str): Type of features ('ECFP' or 'descriptors').

        dist_metric (str): Name of metric to be used to compute distances; can be anything supported by scipy.spatial.distance.pdist.

        task_name (str): Abbreviated name to describe dataset in plot title.

        metric_kwargs: Additional arguments to pass to metric.

    Returns:
        np.ndarray: Distance matrix.

    """
    log = logging.getLogger('ATOM')
    num_cmpds = dataset.X.shape[0]
    if num_cmpds > 50000:
        log.warning("Dataset has %d compounds, too big to calculate distance matrix" % num_cmpds)
        return
    log.warning("Starting distance matrix calculation for %d compounds" % num_cmpds)
    dists = cd.calc_dist_diskdataset(feat_type, dist_metric, dataset, calc_type='all', **metric_kwargs)
    log.warning("Finished calculation of %d distances" % len(dists))
    if len(dists) > ndist_max:
        # Sample a subset of the distances so KDE doesn't take so long
        dist_sample = np.random.choice(dists, size=ndist_max)
    else:
        dist_sample = dists

    dist_pdf = gaussian_kde(dist_sample)
    x_plt = np.linspace(min(dist_sample), max(dist_sample), 500)
    y_plt = dist_pdf(x_plt)
    fig, ax = plt.subplots(figsize=(8.0,8.0))
    ax.plot(x_plt, y_plt, color='forestgreen')
    ax.set_xlabel('%s distance' % dist_metric)
    ax.set_ylabel('Density')
    ax.set_title("%s dataset\nDistribution of %s distances between %s feature vectors" % (
                  task_name, dist_metric, feat_type))
    return dists

#------------------------------------------------------------------------------------------------------------------
def plot_tani_dist_distr(df, smiles_col, df_name, radius=2, subset_col='subset', subsets=False, 
                         ref_subset='train', plot_width=6, ndist_max=None, **metric_kwargs):
    """Generate a density plot showing the distribution of nearest neighbor distances between
    ecfp feature vectors, using the Tanimoto metric. Optionally split by subset.

    Args:
        df (DataFrame): A data frame containing, at minimum, a column of SMILES strings.

        smiles_col (str): Name of the column containing SMILES strings.

        df_name (str): Name for the dataset, to be used in the plot title.

        radius (int): Radius parameter used to calculate ECFP fingerprints. The default is 2, meaning that ECFP4
        fingerprints are calculated.

        subset_col (str): Name of the column containing subset names.

        subsets (bool): If True, distances are only calculated for compounds not in the reference subset, and the
        distances computed are to the nearest neighbors in the reference subset.

        ref_subset (str): Reference subset for nearest-neighbor distances, if `subsets` is True.

        plot_width (float): Plot width in inches.

        ndist_max (int): Not used, included only for backward compatibility.

        metric_kwargs: Additional arguments to pass to metric. Not used, included only for backward compatibility.

    Returns:
        dist (DataFrame): Table of individual nearest-neighbor Tanimoto distance values. If subsets is True,
        the table will include a column indicating the subset each compound belongs to.

    """
    log = logging.getLogger('ATOM')
    num_cmpds = len(df)
    # TODO: Make max compounds a parameter, rather than hardcoding to 50000. Better yet, calculate a sample
    # of distances of size ndist_max for each non-reference subset, and plot KDEs based on the samples.
    if num_cmpds > 50000:
        log.warning("Dataset has %d compounds, too big to calculate distance matrix" % num_cmpds)
        return

    if subsets and subset_col not in df.columns:
        log.warning(f"{subset_col} column not found. Calculating total tanimoto distances instead.")
        subsets=False
    feat_type = 'ecfp'
    dist_metric = 'tanimoto'
    if not subsets:
        smiles_arr1 = df[smiles_col].values
        dists=cd.calc_dist_smiles(feat_type, dist_metric, smiles_arr1, calc_type='nearest', num_nearest=1)
#         dists=cd.calc_dist_smiles(feat_type, dist_metric, smiles_arr1, calc_type='all')
        #print(len(smiles_arr1), dists.shape)
        # flatten dists
        dists = dists.flatten()
        subs=['all']*len(dists)
        dists=pd.DataFrame(zip(dists,subs), columns=['dist','subset'])
    elif subsets:
        dists=pd.DataFrame([], columns=['dist','subset'])
        for subs in df[subset_col].unique():
            if subs==ref_subset:
                continue
            smiles_arr1 = df.loc[df[subset_col]==ref_subset, smiles_col].values
            smiles_arr2 = df.loc[df[subset_col]==subs, smiles_col].values
            diststmp = cd.calc_dist_smiles(feat_type, dist_metric, smiles_arr2, smiles_arr2=smiles_arr1, calc_type='nearest', num_nearest=1)
#             diststmp = cd.calc_dist_smiles(feat_type, dist_metric, smiles_arr2, smiles_arr2=smiles_arr1, calc_type='all')
            #print(subs, diststmp.shape)
            # flatten dists
            diststmp = diststmp.flatten()
            substmp=[subs]*len(diststmp)
            diststmp = pd.DataFrame(zip(diststmp,substmp), columns=['dist','subset'])
            dists=pd.concat([dists,diststmp])
    dists=dists.reset_index(drop=True)
    fig, ax = plt.subplots(1, figsize=(plot_width, plot_width), dpi=300)
    sns.kdeplot(data=dists[dists.subset!=ref_subset], x='dist', hue='subset', legend=True, common_norm=False, common_grid=True, fill=False, ax=ax)
    ax.set_xlabel('%s distance' % dist_metric)
    ax.set_ylabel('Density')
    if not subsets:
        ax.set_title("%s dataset\nDistribution of %s nearest neighbor distances between %s feature vectors" % (
                      df_name, dist_metric, feat_type))
    else: 
        ax.set_title(f"{df_name} dataset: Distribution of {dist_metric} distances\nbetween {feat_type} feature vectors from non-{ref_subset} subsets\nto their nearest neighbors in the {ref_subset} subset")

    return dists

#------------------------------------------------------------------------------------------------------------------
def diversity_plots(dset_key, datastore=True, bucket='public', title_prefix=None, ecfp_radius=4, umap_file=None, out_dir=None,
                    id_col='compound_id', smiles_col='rdkit_smiles', is_base_smiles=False, response_col=None, max_for_mcs=300, colorpal=None):
    """Plot visualizations of diversity for an arbitrary table of compounds. At minimum, the file should contain
    columns for a compound ID and a SMILES string. Produces a clustered heatmap display of Tanimoto distances between
    compounds along with a 2D UMAP projection plot based on ECFP fingerprints, with points colored according to the response
    variable.

    Args:
        dset_key (str): Datastore key or filepath for dataset.

        datastore (bool): Whether to load dataset from datastore or from filesystem.

        bucket (str): Name of datastore bucket containing dataset.

        title_prefix (str): Prefix for plot titles.

        ecfp_radius (int): Radius for ECFP fingerprint calculation.

        umap_file (str, optional): Path to file to write UMAP coordinates to.

        out_dir (str, optional):  Output directory for plots and tables. If provided, plots will be output as PDF files rather
            than in the current notebook, and some additional CSV files will be generated.

        id_col (str): Column in dataset containing compound IDs.

        smiles_col (str): Column in dataset containing SMILES strings.

        is_base_smiles (bool): True if SMILES strings do not need to be salt-stripped and standardized.

        response_col (str): Column in dataset containing response values.

        max_for_mcs (int): Maximum dataset size for plots based on MCS distance. If the number of compounds is less than this
            value, an additional cluster heatmap and UMAP projection plot will be produced based on maximum common substructure
            distance.

    """
    # Load table of compound names, IDs and SMILES strings
    if datastore:
        cmpd_df = dsf.retrieve_dataset_by_datasetkey(dset_key, bucket)
    else:
        cmpd_df = pd.read_csv(dset_key, index_col=False)
    cmpd_df = cmpd_df.drop_duplicates(subset=smiles_col)
    file_prefix = os.path.splitext(os.path.basename(dset_key))[0]
    if title_prefix is None:
        title_prefix = file_prefix.replace('_', ' ')
    compound_ids = cmpd_df[id_col].values
    smiles_strs = cmpd_df[smiles_col].values
    ncmpds = len(smiles_strs)
    # Strip salts, canonicalize SMILES strings and create RDKit Mol objects
    if is_base_smiles:
        base_mols = np.array([Chem.MolFromSmiles(s) for s in smiles_strs])
    else:
        print("Canonicalizing %d molecules..." % ncmpds)
        base_mols = np.array([struct_utils.base_mol_from_smiles(smiles) for smiles in smiles_strs])
        for i, mol in enumerate(base_mols):
            if mol is None:
                print('Unable to get base molecule for compound %d = %s' % (i, compound_ids[i]))
        print("Done")

    has_good_smiles = np.array([mol is not None for mol in base_mols])
    base_mols = base_mols[has_good_smiles]

    cmpd_df = cmpd_df[has_good_smiles]
    ncmpds = cmpd_df.shape[0]
    compound_ids = cmpd_df[id_col].values
    responses = None
    if response_col is not None:
        responses = cmpd_df[response_col].values
        uniq_responses = set(responses)
        if colorpal is None:
            if uniq_responses == set([0,1]):
                _response_type = 'binary'
                colorpal =  {0 : 'forestgreen', 1 : 'red'}
            elif len(uniq_responses) <= 10:
                _response_type = 'categorical'
                colorpal = sns.color_palette('husl', n_colors=len(uniq_responses))
            else:
                _response_type = 'continuous'
                colorpal = sns.blend_palette(['red', 'green', 'blue'], 12, as_cmap=True)



    # Generate ECFP fingerprints
    print("Computing fingerprints...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, ecfp_radius, 1024) for mol in base_mols if mol is not None]
    print("Done")

    if ncmpds <= max_for_mcs:
        # Get MCS distance matrix and draw a heatmap
        print("Computing MCS distance matrix...")
        mcs_dist = dm.mcs(base_mols)
        print("Done")
        cmpd1 = []
        cmpd2 = []
        dist = []
        ind1 = []
        ind2 = []
        for i in range(ncmpds-1):
            for j in range(i+1, ncmpds):
                cmpd1.append(compound_ids[i])
                cmpd2.append(compound_ids[j])
                dist.append(mcs_dist[i,j])
                ind1.append(i)
                ind2.append(j)
        dist_df = pd.DataFrame({'compound_1' : cmpd1, 'compound_2' : cmpd2, 'dist' : dist,
                                'i' : ind1, 'j' : ind2})
        dist_df = dist_df.sort_values(by='dist')
        print(dist_df.head(10))
        if out_dir is not None:
            dist_df.to_csv('%s/%s_mcs_dist_table.csv' % (out_dir, file_prefix), index=False)
            for k in range(10):
                mol_i = base_mols[dist_df.i.values[k]]
                mol_j = base_mols[dist_df.j.values[k]]
                img_file_i = '%s/%d_%s.png' % (out_dir, k, compound_ids[dist_df.i.values[k]])
                img_file_j = '%s/%d_%s.png' % (out_dir, k, compound_ids[dist_df.j.values[k]])
                Draw.MolToFile(mol_i, img_file_i, size=(500,500), fitImage=False)
                Draw.MolToFile(mol_j, img_file_j, size=(500,500), fitImage=False)
    
        mcs_linkage = linkage(mcs_dist, method='complete')
        mcs_df = pd.DataFrame(mcs_dist, columns=compound_ids, index=compound_ids)
        if out_dir is not None:
            pdf_path = '%s/%s_mcs_clustermap.pdf' % (out_dir, file_prefix)
            pdf = PdfPages(pdf_path)
        g = sns.clustermap(mcs_df, row_linkage=mcs_linkage, col_linkage=mcs_linkage, figsize=(12,12), cmap='plasma')
        if out_dir is not None:
            pdf.savefig(g.fig)
            pdf.close()
    
        # Draw a UMAP projection based on MCS distance
        mapper = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, metric='precomputed', random_state=17)
        reps = mapper.fit_transform(mcs_dist)
        rep_df = pd.DataFrame.from_records(reps, columns=['x', 'y'])
        rep_df['compound_id'] = compound_ids
        if out_dir is not None:
            pdf_path = '%s/%s_mcs_umap_proj.pdf' % (out_dir, file_prefix)
            pdf = PdfPages(pdf_path)
        fig, ax = plt.subplots(figsize=(12,12))
        if responses is None:
            sns.scatterplot(x='x', y='y', data=rep_df, ax=ax)
        else:
            rep_df['response'] = responses
            sns.scatterplot(x='x', y='y', hue='response', palette=colorpal,
                            data=rep_df, ax=ax)
        ax.set_title("%s, 2D projection based on MCS distance" % title_prefix)
        if out_dir is not None:
            pdf.savefig(fig)
            pdf.close()
            rep_df.to_csv('%s/%s_mcs_umap_proj.csv' % (out_dir, file_prefix), index=False)

    # Get Tanimoto distance matrix
    print("Computing Tanimoto distance matrix...")
    tani_dist = dm.tanimoto(fps)
    print("Done")
    # Draw a UMAP projection based on Tanimoto distance
    mapper = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, metric='precomputed', random_state=17)
    reps = mapper.fit_transform(tani_dist)
    rep_df = pd.DataFrame.from_records(reps, columns=['x', 'y'])
    rep_df['compound_id'] = compound_ids
    if responses is not None:
        rep_df['response'] = responses
    if umap_file is not None:
        rep_df.to_csv(umap_file, index=False)
        print("Wrote UMAP mapping to %s" % umap_file)
    if out_dir is not None:
        pdf_path = '%s/%s_tani_umap_proj.pdf' % (out_dir, file_prefix)
        pdf = PdfPages(pdf_path)
    fig, ax = plt.subplots(figsize=(12,12))
    if responses is None:
        sns.scatterplot(x='x', y='y', data=rep_df, ax=ax)
    else:
        sns.scatterplot(x='x', y='y', hue='response', palette=colorpal,
                        data=rep_df, ax=ax)
    ax.set_title("%s, 2D projection based on Tanimoto distance" % title_prefix)
    if out_dir is not None:
        pdf.savefig(fig)
        pdf.close()

    # Draw a cluster heatmap based on Tanimoto distance
    tani_linkage = linkage(tani_dist, method='complete')
    tani_df = pd.DataFrame(tani_dist, columns=compound_ids, index=compound_ids)
    if out_dir is not None:
        pdf_path = '%s/%s_tanimoto_clustermap.pdf' % (out_dir, file_prefix)
        pdf = PdfPages(pdf_path)
    g = sns.clustermap(tani_df, row_linkage=tani_linkage, col_linkage=tani_linkage, figsize=(12,12), cmap='plasma')
    if out_dir is not None:
        pdf.savefig(g.fig)
        pdf.close()

