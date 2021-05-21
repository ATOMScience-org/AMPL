"""Distance metrics for compounds: Tanimoto and maximum common substructure (MCS)"""

import itertools
import multiprocessing

from rdkit import DataStructs
from rdkit.Chem.rdFMCS import FindMCS  # pylint: disable=no-name-in-module
from scipy.spatial.distance import squareform  # pylint: disable=unused-import
import numpy as np
N_PROCS = 1 #multiprocessing.cpu_count()


def parallel_dist_single(inp_lst, worker_fn):
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

def parallel_dist_multi(inp_lst, worker_fn):
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

def tanimoto_worker(k, fps):
    """Get per-fingerprint Tanimoto distance vector."""
    # pylint: disable=no-member
    sims = DataStructs.BulkTanimotoSimilarity(fps[k], fps[(k + 1):])
    dists_k = [1. - s for s in sims]
    return np.array(dists_k), 0

def tanimoto_single(fp, fps):
    """Get per-fingerprint Tanimoto distance vector."""
    # pylint: disable=no-member
    sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
    dists = [1. - s for s in sims]
    return np.array(dists), 0

def tanimoto(fps1, fps2=None):
    """Compute Tanimoto distance between given ECFP fingerprints."""
    if fps2 is None:
        dists, _ = parallel_dist_single([fps1], tanimoto_worker)
    else:
        dists, _ = parallel_dist_multi([fps1, fps2], tanimoto_single)
    return dists


def mcs_worker(k, mols, n_atms):
    """Get per-molecule MCS distance vector."""
    dists_k = []
    n_incomp = 0  # Number of searches terminated before timeout
    for l in range(k + 1, len(mols)):
        # Set timeout to halt exhaustive search, which could take minutes
        result = FindMCS([mols[k], mols[l]], completeRingsOnly=True,
                         ringMatchesRingOnly=True, timeout=10)
        dists_k.append(1. - result.numAtoms /
                       ((n_atms[k] + n_atms[l]) / 2))
        if result.canceled:
            n_incomp += 1
    return np.array(dists_k), n_incomp

def mcs_single(mol, mols, n_atms):
    """Get per-molecule MCS distance vector."""
    dists_k = []
    n_atm = float(mol.GetNumAtoms())
    n_incomp = 0  # Number of searches terminated before timeout
    for l in range(0, len(mols)):
        # Set timeout to halt exhaustive search, which could take minutes
        result = FindMCS([mol, mols[l]], completeRingsOnly=True,
                         ringMatchesRingOnly=True, timeout=10)
        dists_k.append(1. - result.numAtoms /
                       ((n_atm + n_atms[l]) / 2))
        if result.canceled:
            n_incomp += 1
    return np.array(dists_k), n_incomp

def mcs(mols1, mols2=None):
    """Compute average variant of MCS distance between molecules."""
    n_atms1 = [float(m.GetNumAtoms()) for m in mols1]
    if mols2 is None:
        dists, sum_incomplete = parallel_dist_single([mols1, n_atms1], mcs_worker)
    else:
        dists, sum_incomplete = parallel_dist_multi([mols1, mols2, n_atms1], mcs_single)
    if sum_incomplete:
        print('{} incomplete MCS searches'.format(sum_incomplete))
    return dists


if __name__ == '__main__':
    # Start a server process, which parent forks for child processes
    multiprocessing.set_start_method('forkserver')
