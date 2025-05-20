"""Functions to generate matrices or vectors of distances between compounds"""

import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import pandas as pd

from atomsci.ddm.pipeline import dist_metrics

def calc_dist_smiles(feat_type, dist_met, smiles_arr1, smiles_arr2=None, calc_type='nearest', num_nearest=1, **metric_kwargs):
    """Returns an array of distances between compounds given as SMILES strings, either between all pairs of compounds in a
    single dataset or between two datasets.

    Args:
        feat_type (str): How the data is to be featurized, if dist_met is not 'mcs'. The only option supported currently is 'ECFP'.

        dist_met (str): What distance metric to use. Current options include 'tanimoto' and 'mcs'.

        smiles_arr1 (list): First list of SMILES strings.

        smiles_arr2 (list): Optional, second list of SMILES strings. Can have only 1 member if wanting compound to
        matrix comparison.

        calc_type (str): Type of summarization to perform on rows of distance matrix. See function calc_summary for options.

        num_nearest (int): Additional parameter for calc_types nearest, nth_nearest and avg_n_nearest.

        metric_kwargs: Additional arguments to be passed to functions that calculate metrics.

    Returns:
        dists: vector or array of distances

    Todo:
        Provide an option to compute distances based on descriptor values.

        Allow other metrics for ECFP features, as in calc_dist_diskdataset().

    """
    within_dset = False
    if feat_type in ['ECFP','ecfp'] and dist_met=='tanimoto':
        mols1 = [Chem.MolFromSmiles(s) for s in smiles_arr1]
        fprints1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols1]
        if smiles_arr2 is not None:
            if len(smiles_arr2) == 1:
                cpd_mol = Chem.MolFromSmiles(smiles_arr2[0])
                cpd_fprint = AllChem.GetMorganFingerprintAsBitVect(cpd_mol, 2, 1024)
                # Vector of distances
                return calc_summary(dist_metrics.tanimoto_single(cpd_fprint, fprints1)[0], calc_type, 
                                    num_nearest, within_dset)
            else:
                mols2 = [Chem.MolFromSmiles(s) for s in smiles_arr2]
                fprints2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols2]
        else:
            fprints2 = None
            within_dset = True
        return calc_summary(dist_metrics.tanimoto(fprints1, fprints2), calc_type, num_nearest, within_dset)
        
    elif dist_met == 'mcs':
        mols1 = [Chem.MolFromSmiles(s) for s in smiles_arr1]
        n_atms = [mol.GetNumAtoms() for mol in mols1]
        if smiles_arr2 is not None:
            if len(smiles_arr2) == 1:
                cpd_mol = Chem.MolFromSmiles(smiles_arr2[0])
                # Vector of distances
                return calc_summary(dist_metrics.mcs_single(
                    cpd_mol, mols1, n_atms)[0], calc_type, num_nearest, within_dset)
            else:
                mols2 = [Chem.MolFromSmiles(s) for s in smiles_arr2]
        else:
            mols2 = None
        return calc_summary(dist_metrics.mcs(mols1, mols2), calc_type, num_nearest, within_dset=True)
    
    elif feat_type in ['descriptors', 'moe']:
        raise ValueError("Descriptor features are not currently supported by calc_dist_smiles().")



def calc_dist_diskdataset(feat_type, dist_met, dataset1, dataset2=None, calc_type='nearest', num_nearest=1, **metric_kwargs):
    """Returns an array of distances, either between all compounds in a single dataset or between two datasets, given
    as DeepChem Dataset objects.

    Args:
        feat_type (str): How the data was featurized. Current options are 'ECFP' or 'descriptors'.

        dist_met (str): What distance metric to use. Current options include tanimoto, cosine, cityblock, euclidean, or any
        other metric supported by scipy.spatial.distance.pdist().

        dataset1 (deepchem.Dataset): Dataset containing features of compounds to be compared.

        dataset2 (deepchem.Dataset, optional): Second dataset, if two datasets are to be compared.

        calc_type (str): Type of summarization to perform on rows of distance matrix. See function calc_summary for options.

        num_nearest (int): Additional parameter for calc_types nearest, nth_nearest and avg_n_nearest.

        metric_kwargs: Additional arguments to be passed to functions that calculate metrics.

    Returns:
        np.ndarray: Vector or matrix of distances between feature vectors.

    """
    if dataset2 is not None:
        return calc_dist_feat_array(feat_type, dist_met, dataset1.X, dataset2.X, calc_type, num_nearest, **metric_kwargs)
    else:
        return calc_dist_feat_array(feat_type, dist_met, dataset1.X, None, calc_type, num_nearest, **metric_kwargs)
    
def calc_dist_feat_array(feat_type, dist_met, feat1, feat2=None, calc_type='nearest', num_nearest=1, **metric_kwargs):
    """Returns a vector or array of distances, either between all compounds in a single dataset or between two datasets,
    given the feature matrices for the dataset(s).

    Args:
        feat_type (str): How the data was featurized. Current options are 'ECFP' or 'descriptors'.

        dist_met (str): What distance metric to use. Current options include tanimoto, cosine, cityblock, euclidean, or any
        other metric supported by scipy.spatial.distance.pdist().

        feat1: feature matrix as a numpy array

        feat2: Optional, second feature matrix

        calc_type (str): Type of summarization to perform on rows of distance matrix. See function calc_summary for options.

        num_nearest (int): Additional parameter for calc_types nearest, nth_nearest and avg_n_nearest.

        metric_kwargs: Additional arguments to be passed to functions that calculate metrics.

    Returns:
        dists: vector or array of distances

    """

    if feat_type in ['ECFP', 'ecfp']:
        if dist_met == 'tanimoto':
            if feat2 is not None:
                if feat2.shape[0] == 1:
                    # Vector of distances
                    return calc_summary(dist_metrics.tanimoto_single(feat2, feat1)[0], calc_type, 
                                        num_nearest)
                return calc_summary(dist_metrics.tanimoto(feat1, feat2), calc_type, num_nearest)
            else:
                return calc_summary(dist_metrics.tanimoto(feat1), calc_type, num_nearest, within_dset=True)
        else:
            if feat2 is not None:
                return calc_summary(cdist(feat1, feat2, dist_met), calc_type, num_nearest)
            return calc_summary(pdist(feat1, dist_met, **metric_kwargs), calc_type, num_nearest, within_dset=True)

    elif feat_type == 'descriptors':
        if feat2 is not None:
            return calc_summary(cdist(feat1, feat2, dist_met), calc_type, num_nearest)
        return calc_summary(pdist(feat1, dist_met, **metric_kwargs), calc_type, num_nearest, within_dset=True)
    


def calc_summary(dist_arr, calc_type, num_nearest=1, within_dset=False):
    """Returns a summary of the distances in dist_arr, depending on calc_type.

    Args:
        dist_arr: (np.array): Either a 2D distance matrix, or a 1D condensed distance matrix (flattened upper triangle).

        calc_type (str): The type of summary values to return:

            all:            The distance matrix itself

            nearest:        The distances to the num_nearest nearest neighbors of each compound (except compound itself)

            nth_nearest:    The distance to the num_nearest'th nearest neighbor

            avg_n_nearest:  The average of the num_nearest nearest neighbor distances

            farthest:       The distance to the farthest neighbor

            avg:            The average of all distances for each compound

        num_nearest (int):  Additional parameter for calc_types nearest, nth_nearest and avg_n_nearest.

        within_dset (bool): True if input distances are between compounds in the same dataset.
        
    Returns:
        dists (np.array):  A numpy array of distances. For calc_type 'nearest' with num_nearest > 1, this is a 2D array
        with a row for each compound; otherwise it is a 1D array.
    """

    if calc_type == 'all':
        return dist_arr

    if len(dist_arr.shape) == 1:
        dist_mat = squareform(dist_arr)
    else:
        dist_mat = dist_arr

    if calc_type == 'farthest':
        return dist_mat.max(axis=1)
    if calc_type == 'avg':
        return dist_mat.mean(axis=1)
    if calc_type == 'nearest':
        nn_dist = np.sort(dist_mat)
        if within_dset:
            # Exclude the zero distances between each compound and itself. But don't exclude
            # zero distances between different compounds!
            nn_dist = nn_dist[:,1:(num_nearest+1)]
        else:
            nn_dist = nn_dist[:,:num_nearest]
        if num_nearest == 1:
            return nn_dist[:,0]
        else:
            return nn_dist

    if calc_type == 'nth_nearest':
        nn_dist = np.sort(dist_mat)
        if within_dset:
            return nn_dist[:,num_nearest]
        else:
            return nn_dist[:,num_nearest-1]

    if calc_type == 'avg_n_nearest':
        if within_dset:
            return np.sort(dist_mat)[:,1:(num_nearest+1)].mean(axis=1)
        else:
            return np.sort(dist_mat)[:,:num_nearest].mean(axis=1)

    else:
        print("calc_type %s is not valid" % calc_type)
        sys.exit(1)
        
def upload_distmatrix_to_DS(
        dist_matrix,feature_type,compound_ids,bucket,title,description,tags,key_values,filepath="./",dataset_key=None):
    """Uploads distance matrix in the data store with the appropriate tags

    Args:
       dist_matrix (np.ndarray): The distance matrix.

       feature_type (str): How the data was featurized.

       dist_met (str): What distance metric was used.

       compound_ids (list): list of compound ids corresponding to the distance matrix (assumes that distance matrix is square
       and is the distance between all compounds in a dataset)

       bucket (str): bucket the file will be put in

       title (str): title of the file in (human friendly format)

       description (str): long text box to describe file (background/use notes)

       tags (list): List of tags to assign to datastore object.

       key_values (dict): Dictionary of key:value pairs to include in the datastore object's metadata.

       filepath (str): local path where you want to store the pickled dataframe

       dataset_key (str): If updating a file already in the datastore enter the corresponding dataset_key.
                     If not, leave as 'none' and the dataset_key will be automatically generated.

    Returns:
        None
    """
    from atomsci.ddm.utils import datastore_functions as dsf
    fnm = "distmatrix_nm"
    dist_df = pd.DataFrame(dist_matrix)
    dist_df.index = compound_ids
    dist_df.columns = compound_ids
    filename = fnm.replace("nm",feature_type) # fn is not defined anywhere. need to address this
    _dist_pkl = dist_df.to_pickle(filepath + filename)
    dsf.upload_file_to_DS(bucket, title, description, tags, key_values, filepath, filename, dataset_key, client=None)

