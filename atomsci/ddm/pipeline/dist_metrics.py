"""Distance metrics for compounds: Tanimoto and maximum common substructure (MCS)"""

import itertools
import multiprocessing

from rdkit import DataStructs
from rdkit.Chem.rdFMCS import FindMCS  # pylint: disable=no-name-in-module
from scipy.spatial.distance import squareform  # pylint: disable=unused-import
import numpy as np
N_PROCS = multiprocessing.cpu_count()


def _parallel_dist_single(inp_lst, worker_fn):
    """Method for multiprocessor distance matrix computation."""
    pool = multiprocessing.Pool(processes=N_PROCS)
    inputs = [[k] + inp_lst for k, _ in enumerate(inp_lst[0])]
    ret = pool.starmap(worker_fn, inputs)
    pool.close()
    pool.join()

    dists_all = []
    sum_incomp = 0
    for r in ret:
        dists_all.append(r[0])
        sum_incomp += r[1]

    # Flattened distance matrix (upper triangle only)
    dists = np.array(list(itertools.chain.from_iterable(dists_all)))
    return squareform(dists), sum_incomp

def _parallel_dist_multi(inp_lst, worker_fn):
    """Method for multiprocessor distance matrix computation."""
    pool = multiprocessing.Pool(processes=N_PROCS)
    #TODO: Want to switch order of fps1 and 2?
    inputs = [[inp_lst[0][k]] + inp_lst[1:] for k, _ in enumerate(inp_lst[0])]
    ret = pool.starmap(worker_fn, inputs)
    pool.close()
    pool.join()

    dists_all = []
    sum_incomp = 0
    for r in ret:
        dists_all.append(r[0])
        sum_incomp += r[1]
    return np.asarray(dists_all), sum_incomp

def _tanimoto_worker(k, fps):
    """Get per-fingerprint Tanimoto distance vector."""
    # pylint: disable=no-member
    sims = DataStructs.BulkTanimotoSimilarity(fps[k], fps[(k + 1):])
    dists_k = [1. - s for s in sims]
    return np.array(dists_k), 0

def tanimoto_single(fp, fps):
    """Compute a vector of Tanimoto distances between a single fingerprint and each fingerprint in a list .

    Args:
        fp : Fingerprint to be compared.

        fps (Sequence): List of ECFP fingerprint vectors.

    Returns:
        np.ndarray: Vector of distances between fp and each fingerprint in fps.

    """
    # pylint: disable=no-member
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    dists = [1. - s for s in sims]
    return np.array(dists), 0

def tanimoto(fps1, fps2=None):
    """Compute Tanimoto distances between sets of ECFP fingerprints.

    Args:
        fps1 (Sequence): First list of ECFP fingerprint vectors.

        fps2 (Sequence, optional): Second list of ECFP fingerprint vectors.
            If not provided, computes distances between pairs of fingerprints in fps1.
            Otherwise, computes a matrix of distances between pairs of fingerprints in fps1 and fps2.

    Returns:
        np.ndarray: Matrix of pairwise distances between fingerprints.
    """
    if fps2 is None:
        dists, _ = _parallel_dist_single([fps1], _tanimoto_worker)
    else:
        dists, _ = _parallel_dist_multi([fps1, fps2], tanimoto_single)
    return dists


def _mcs_worker(k, mols, n_atms):
    """Get per-molecule MCS distance vector."""
    dists_k = []
    n_incomp = 0  # Number of searches terminated before timeout
    for idx in range(k + 1, len(mols)):
        # Set timeout to halt exhaustive search, which could take minutes
        result = FindMCS([mols[k], mols[idx]], completeRingsOnly=True,
                         ringMatchesRingOnly=True, timeout=10)
        dists_k.append(1. - result.numAtoms /
                       ((n_atms[k] + n_atms[idx]) / 2))
        if result.canceled:
            n_incomp += 1
    return np.array(dists_k), n_incomp

def _mcs_single(mol, mols, n_atms):
    """Get per-molecule MCS distance vector."""
    dists_k = []
    n_atm = float(mol.GetNumAtoms())
    n_incomp = 0  # Number of searches terminated before timeout
    for idx in range(0, len(mols)):
        # Set timeout to halt exhaustive search, which could take minutes
        result = FindMCS([mol, mols[idx]], completeRingsOnly=True,
                         ringMatchesRingOnly=True, timeout=10)
        dists_k.append(1. - result.numAtoms /
                       ((n_atm + n_atms[idx]) / 2))
        if result.canceled:
            n_incomp += 1
    return np.array(dists_k), n_incomp

def mcs(mols1, mols2=None):
    """Computes maximum common substructure (MCS) distances between pairs of molecules.

    The MCS distance between molecules m1 and m2 is one minus the average of fMCS(m1,m2) and fMCS(m2,m1),
    where fMCS(m1,m2) is the fraction of m1's atoms that are part of the largest common substructure of m1 and m2.

    Args:
        mols1 (Sequence of `rdkit.Mol`): First list of molecules.

        mols2 (Sequence of `rdkit.Mol`, optional): Second list of molecules.
            If not provided, computes MCS distances between pairs of molecules in mols1.
            Otherwise, computes a matrix of distances between pairs of molecules from mols1 and mols2.

    Returns:
        np.ndarray: Matrix of pairwise distances between molecules.

    """

    n_atms1 = [float(m.GetNumAtoms()) for m in mols1]
    if mols2 is None:
        dists, sum_incomplete = _parallel_dist_single([mols1, n_atms1], _mcs_worker)
    else:
        dists, sum_incomplete = _parallel_dist_multi([mols1, mols2, n_atms1], _mcs_single)
    if sum_incomplete:
        print('{} incomplete MCS searches'.format(sum_incomplete))
    return dists


if __name__ == '__main__':
    # Start a server process, which parent forks for child processes
    multiprocessing.set_start_method('forkserver')
