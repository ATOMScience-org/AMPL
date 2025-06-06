import argparse
import logging
import tempfile
from typing import List, Optional, Set, Tuple
from functools import partial
import numpy as np
import pandas as pd
from scipy import stats

import deepchem as dc
from deepchem.data import Dataset
from deepchem.splits import Splitter
from deepchem.splits.splitters import _generate_scaffold

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from atomsci.ddm.pipeline import chem_diversity as cd
from atomsci.ddm.pipeline import dist_metrics
from atomsci.ddm.pipeline import GeneticAlgorithm as ga

logging.basicConfig(format='%(asctime)-15s %(message)s')
logger = logging.getLogger('ATOM')

def _generate_scaffold_hists(scaffold_sets: List[np.ndarray], 
                                w: np.array) -> np.array:
    """Counts the number of labelled samples per task per scaffold

    Returns an np.array M where each row i represents a scaffold and
    each column j represents a task and M[i,j] contains the number of
    labelled examples scaffold j has for task i.

    Parameters
    ----------
    scaffold_sets: List[np.ndarray]
        A list of scaffolds. Each scaffold is an array of compound indices.

    w: np.array
        This is the w member of a Dataset. It is a binary matrix
        denoting if compound i has a label for task j.

    Returns
    -------
    scaffold_hists: np.array
        An np.array M where each row i represents a scaffold and
        each column j represents a task and M[i,j] contains the number of
        labelled examples scaffold j has for task i.

    """
    scaffold_hists = np.zeros((len(scaffold_sets), w.shape[1]), int)
    for i, scaffold_set in enumerate(scaffold_sets):
        scaffold_hists[i] = np.sum(w[list(scaffold_set)], axis=0)

    return scaffold_hists

def smush_small_scaffolds(scaffolds: List[Set[int]], 
                            num_super_scaffolds: int = 100) -> List[np.ndarray]:
    """Combines small scaffolds into super scaffolds

    Since using Murcko scaffolds
    usually results in mostly scaffolds with 1 compound, these are 'super scaffolds'.
    Each of these scaffolds are made up of Murcko scaffolds. Murcko scaffolds
    are combined using the same method as ScaffoldSplitter, just extended to make
    n 'super scaffolds'.

    Parameters
    ----------
    scaffolds: List[Set[int]]
        A list of scaffolds

    num_super_scaffolds: int
        The number of desired super scaffolds

    Returns
    -------
    new_scaffolds: List[np.ndarray]
        A list of super scaffolds. All roughly the same size. Unless the original
        list of scaffolds is shorter than the desired number of scaffolds

    """

    if len(scaffolds) <= num_super_scaffolds:
        # Nothing to do, except change the sets to numpy arrays
        return [np.array(list(s)) for s in scaffolds]

    total_length = np.sum([len(s) for s in scaffolds])
    size_per_scaffold = int(total_length)/(num_super_scaffolds-1)

    new_scaffolds = [set()]
    for scaffold in scaffolds:
        current_scaffold = new_scaffolds[-1]
        if ((len(current_scaffold) + len(scaffold)) < size_per_scaffold) or (len(current_scaffold) == 0):
            current_scaffold.update(scaffold)
        else:
            new_scaffolds.append(scaffold)

    logger.debug('new scaffold lengths')
    logger.debug([len(s) for s in new_scaffolds])
    new_scaffolds = [np.array(list(s)) for s in new_scaffolds]
    return new_scaffolds

def calc_ecfp(smiles: List[str],
                workers: int = 8) -> List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]:
    """Giving a list of strings return a list of ecfp features

    Calls AllChem.GetMorganFingerprintAsBitVect for each smiles in parallel

    Paramters
    ---------
    smiles: List[str]
        List of smiles strings

    workers: int
        Number of parallel workers

    Returns
    -------
    fprints: List[int] TODO, UPDATE WITH REAL TYPE
        Actually this is a special data type used by rdkit, some kind of
        UIntSparseIntVect. This datatype is used specifically with
        dist_smiles_from_ecfp
    """
    func = partial(calc_ecfp,workers=1)
    if workers > 1:
      from multiprocessing import pool
      batchsize = 200
      batches = [smiles[i:i+batchsize] for i in range(0, len(smiles), batchsize)]
      with pool.Pool(workers) as p:
        ecfps = p.map(func,batches)
        fprints = [y for x in ecfps for y in x] #Flatten results
    else:
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024) for mol in mols]

    return fprints

def dist_smiles_from_ecfp(ecfp1: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect],
                            ecfp2: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]) -> List[float]:
    """Calculate tanimoto distance distribution between two lists of ecpf features

    Parameters
    ----------
    ecfp1: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect],
        A list of ECFP fingerprints
    ecfp2: List[rdkit.DataStructs.cDataStructs.ExplicitBitVect]
        A list of ECFP fingerprints

    Returns
    -------
    List[float]
        A list of tanimoto distances between 0 and 1

    """
    if len(ecfp1) == 0 or len(ecfp2) == 0:
        pass
    return cd.calc_summary(dist_metrics.tanimoto(ecfp1, ecfp2), calc_type='nearest', 
                        num_nearest=1, within_dset=False)

class MultitaskScaffoldSplitter(Splitter):
    """MultitaskScaffoldSplitter Splitter class.

    Tries to perform scaffold split across multiple tasks while maintianing
    training, validation, and test fractions for each class using a GeneticAlgorithm

    self.ss: List[np.ndarray]
        Contains a list of arrays of compound indices. Since using Murcko scaffolds
        usually results in mostly scaffolds with 1 compound, these are 'super scaffolds'.
        Each of these scaffolds are made up of Murcko scaffolds. Murcko scaffolds
        are combined using the same method as ScaffoldSplitter, just extended to make
        n 'super scaffolds'. Therefore it is possible to get very close to having the
        ScaffoldSplitter result, if that is the best split possible.

    """

    def generate_scaffolds(self,
                            dataset: Dataset) -> List[Set[int]]:
        """Returns all scaffolds from the dataset.

        Parameters
        ----------
        dataset: Dataset
          Dataset to be split.

        Returns
        -------
        scaffold_sets: List[Set[int]]
          List of indices of each scaffold in the dataset.
        """
        scaffolds = {}
        _data_len = len(dataset)

        for ind, smiles in enumerate(dataset.ids):
            scaffold = _generate_scaffold(smiles)
            if scaffold is None:
                continue
            if scaffold not in scaffolds:
                scaffolds[scaffold] = {ind}
            else:
                scaffolds[scaffold].add(ind)

        # Sort from largest to smallest scaffold sets
        #scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: len(x[1]), reverse=True)
        ]

        return scaffold_sets

    def _generate_scaffold_dist_matrix(self):
        """Computes matrices used by fitness functions that score splits according
        to the dissimilarity between training and test compound structures. One is 
        a symmetric matrix of nearest neighbor Tanimoto distances between pairs of 
        scaffolds (i.e., between the most similar compounds in the two scaffolds).
        This matrix is used to compute the `scaffold_diff_fitness` function.
        
        The other is a nonsymmetric matrix of boolean vectors `has_nn[i,j]`, of length
        equal to the number of compounds in scaffold `i`, indicating whether each
        compound has a neighbor in scaffold `j` nearer than some threshold Tanimoto 
        distance `dist_thresh`. dist_thresh defaults to 0.3, but could be parameterized.
        The has_nn matrix is used to compute the `far_frac_fitness` function, which is
        a more robust measurement of the train/test set dissimilarity.
        """
        scaffold_lists = self.ss
        ecfp_features = self.ecfp_features

        # First compute the full matrix of distances between all pairs of ECFPs
        dmat = dist_metrics.tanimoto(ecfp_features)

        mat_shape = (len(scaffold_lists), len(scaffold_lists))
        scaff_dist_mat = np.zeros(mat_shape)
        has_near_neighbor_mat = np.empty(mat_shape, dtype=object)

        for i, scaff1 in enumerate(scaffold_lists):
            scaff1_rows = scaff1.reshape((-1,1))
            for j, scaff2 in enumerate(scaffold_lists[:i]):
                if i == j:
                    continue

                scaff1_dists = dmat[scaff1_rows, scaff2]
                min_dist = np.min(scaff1_dists.flatten())

                if min_dist==0:
                    logger.info(f"Scaffolds {i} and {j} have at least one ECFP in common")
                    for k in scaff1:
                        for m in scaff2:
                            if dmat[k,m] == 0:
                                logger.debug(f"\tcompound {k} in scaffold {i}, compound {m} in scaffold {j}")
                                logger.debug(f"\tSMILES {k}: {self.smiles[k]}")
                                logger.debug(f"\tSMILES {m}: {self.smiles[m]}\n")


                scaff_dist_mat[i,j] = min_dist
                scaff_dist_mat[j,i] = min_dist

                # Identify the compounds in scaff1 with a neighbor in scaff2 closer than dist_thresh
                has_near_neighbor_mat[i,j] = np.array(np.min(scaff1_dists, axis=1) < self.dist_thresh)

                # Identify the compounds in scaff2 with a neighbor in scaff1 closer than dist_thresh
                scaff2_rows = scaff2.reshape((-1,1))
                scaff2_dists = dmat[scaff2_rows, scaff1]
                has_near_neighbor_mat[j,i] = np.array(np.min(scaff2_dists, axis=1) < self.dist_thresh)


        self.scaff_scaff_distmat = scaff_dist_mat
        self.has_near_neighbor_mat = has_near_neighbor_mat


    def expand_scaffolds(self,
                        scaffold_list: List[int]) -> List[int]:
        """Turns a list of scaffold indices into a list of compound indices

        Given a list of scaffold indices in self.ss return a list of compound
        indices into self.dataset

        Parameters
        ----------
        scaffold_list List[int]:
            A list of indices into self.ss.

        Returns
        -------
        compound_list List[int]:
            A list of compound indices into dataset
        """
        compound_list = [i for scaffold in scaffold_list for i in self.ss[scaffold]]

        return compound_list

    def split_chromosome_to_compound_split(self, 
                            split_chromosome: List[str]) -> Tuple:
        """Turns a split of scaffolds into a split of compounds

        A chromosome is represented as a list of strings. Each string is
        either 'train', 'valid', or 'test' which means that corresponding
        scaffold belongs in 'train', 'valid', or 'test'

        Parameters
        ----------
        split_chromosome: List[str]
            A list of strings that are either 'train', 'valid', or 'test.

        Returns
        -------
        split: Tuple[int]
            A tuple of length 3. Each element of this tuple contains a list of
        indices into self.dataset. You can use these indices to pick out compounds
        that belong to each partition
        """

        train_part = self.expand_scaffolds([scaff_ind for scaff_ind, part in enumerate(split_chromosome) if part=='train'])
        valid_part = self.expand_scaffolds([scaff_ind for scaff_ind, part in enumerate(split_chromosome) if part=='valid'])
        test_part = self.expand_scaffolds([scaff_ind for scaff_ind, part in enumerate(split_chromosome) if part=='test'])

        split = (train_part, valid_part, test_part)

        return split

    def scaffold_diff_fitness(self, 
                            split_chromosome: List[str],
                            part_a: str,
                            part_b: str) -> float:
        """Grades a chromosome based on how well the partitions are separated

        Grades the quality of the split based on which scaffolds were alloted to
        which partitions. The difference between two partitions (part_a and part_b)
        is measured as the minimum distance between all pairs of scaffolds between
        the two given partitions

        Parameters
        ----------
        scaffold_split: List[str]
            A chromosome is represented as a list of strings. Index i in the
            chromosome contains the partition for scaffold i.

        Returns
        -------
        score: float
            Floating point value beteween 0-1. 1 is the best score and 0 is the worst
        """
        train_scaffolds = [i for i, part in enumerate(split_chromosome) if part==part_a]
        test_scaffolds = [i for i, part in enumerate(split_chromosome) if part==part_b]

        # if a partition is completely empty, return 0
        if len(train_scaffolds) == 0 or len(test_scaffolds) == 0:
            return 0

        min_dist = 1e20
        for ind1 in train_scaffolds:
            for ind2 in test_scaffolds:
                assert(not (ind1 == ind2))
                # use the cached distance matrix to speed up computation
                dist = self.scaff_scaff_distmat[ind1, ind2]
                min_dist = np.min([min_dist, np.min(dist)])

        return min_dist

    def far_frac_fitness(self, 
                            split_chromosome: List[str],
                            train_part: str,
                            test_part: str) -> float:
        """Grades a split according to the fraction of valid/test compounds with
        nearest training set compounds further than some threshold.

        Grades the quality of the split based on which scaffolds were alloted to
        which partitions. The score is measured as the fraction of compounds in
        `test_part` with nearest neighbors in `train_part` at Tanimoto distance
        `self.dist_thresh` or greater.

        Parameters
        ----------
        split_chromosome: List[str]
            A split represented as a list of partition names. Index i in the
            chromosome contains the partition for scaffold i.
        train_part: str
            The name of the partition to be treated as the training subset
        test_part: str
            The name of the partition to be treated as the test subset

        Returns
        -------
        score: float
            Floating point value beteween 0-1. 1 is the best score and 0 is the worst
        """
        # TODO: Eventually, replace strings in chromosome with integers indicating the partition
        # for each scaffold.

        train_scaffolds = [i for i, part in enumerate(split_chromosome) if part==train_part]
        test_scaffolds = [i for i, part in enumerate(split_chromosome) if part==test_part]

        # if a partition is completely empty, return 0
        if len(train_scaffolds) == 0 or len(test_scaffolds) == 0:
            return 0

        # Compute the "far fraction": For each test scaffold S, OR together the boolean vectors
        # from has_near_neighbor_mat for the training scaffolds indicating whether each compound in
        # S has a neighbor in the train scaffold closer than Tanimoto distance dist_thresh. Sum
        # the results over compounds and test scaffolds and divide by the test set size to get the
        # "near fraction"; the far fraction is 1 minus the near fraction.

        near_count = 0
        total_count = 0
        for test_ind in test_scaffolds:
            has_nn = None
            for train_ind in train_scaffolds:
                assert(not (train_ind == test_ind))
                if has_nn is None:
                    has_nn = self.has_near_neighbor_mat[test_ind, train_ind]
                else:
                    has_nn |= self.has_near_neighbor_mat[test_ind, train_ind]
            near_count += sum(has_nn)
            total_count += len(has_nn)

        far_frac = 1 - near_count/total_count
        #print(f"far_frac_fitness: near_count {near_count}, total_count {total_count}, far_frac {far_frac}")
        return far_frac


    def ratio_fitness(self, split_chromosome: List[str]) -> float:
        """Calculates a fitness score based on how well the subset proportions for each task, taking
        only labeled compounds into account, match the proportions requested by the user.

        The score is determined by combining the min, median and max over tasks of the subset fractions
        into a 9-dimensional vector and computing its Euclidean distance from an ideal vector having
        all the fractions as requested.

        Parameters
        ----------
        List[str]: split_chromosome
            A list of strings, index i contains a string, 'train', 'valid', 'test'
            which determines the partition that scaffold belongs

        Returns
        -------
        float
            A float between 0 and 1. 1 best 0 is worst
        """
        # total_counts is the number of labels per task
        total_counts = np.sum(self.dataset.w, axis=0)

        # subset_counts is number of labels per task per subset.
        subset_counts = [np.sum(self.dataset.w[subset], axis=0) for subset in \
                            self.split_chromosome_to_compound_split(split_chromosome)]

        # subset order goes train, valid, test
        subset_ratios = [subset_count/total_counts for subset_count in subset_counts]

        # imagine the perfect split is a point in 9D space. For each subset we measure
        # 3 values, min, median, max. Ideally, these 3 values all equal the desired fraction
        target_split = np.concatenate([[self.frac_train]*3,
                                       [self.frac_valid]*3,
                                       [self.frac_test]*3])

        # this is the current split also in 9D space
        current_split = np.concatenate([[np.min(subset), np.median(subset), np.max(subset)] \
                            for subset in subset_ratios])

        # if any partition is 0, then this split fails
        if min(current_split) == 0:
            return 0
        # worst possible distance to normalize this between 0 and 1
        worst_distance = np.linalg.norm(np.ones(len(target_split)))
        worst_distance = np.sqrt(len(target_split))
        ratio_fit = 1 - np.linalg.norm(target_split-current_split)/worst_distance

        return ratio_fit

    def response_distr_fitness(self, split_chromosome: List[str]) -> float:
        """Calculates a fitness score based on how well the validation and test set response
        value distributions match that of the train subset. We measure the degree of 
        matching using the Wasserstein distance.

        Parameters
        ----------
        List[str]: split_chromosome
            A list of strings, index i contains a string, 'train', 'valid', 'test'
            which determines the partition that scaffold belongs

        Returns
        -------
        float
            One minus the sum of the Wasserstein distances between the valid and test subset and
            the train subset response value distributions, averaged over tasks. One means
            the distributions perfectly match.
        """
        if not self.sanity_check_chromosome(split_chromosome):
            return 0.0
        dist_sum = 0.0
        ntasks = self.dataset.y.shape[1]
        train_ind, valid_ind, test_ind = self.split_chromosome_to_compound_split(split_chromosome)
        for task in range(ntasks):
            train_y = self.dataset.y[train_ind, task]
            train_y = train_y[~np.isnan(train_y)]

            valid_y = self.dataset.y[valid_ind, task]
            valid_y = valid_y[~np.isnan(valid_y)]

            test_y = self.dataset.y[test_ind, task]
            test_y = test_y[~np.isnan(test_y)]

            test_dist = stats.wasserstein_distance(train_y, test_y)
            valid_dist = stats.wasserstein_distance(train_y, valid_y)

            dist_sum += valid_dist + test_dist

        avg_dist = dist_sum/(ntasks*2)
        return 1 - avg_dist

    def sanity_check_chromosome(self, split_chromosome: List[str]) -> bool:
        """Sanity checks a chromosome

        Checks to see that each subset has at least 1 labelled compound

        Parameters
        ----------
        List[str]: split_chromosome
            A list of strings, index i contains a string, 'train', 'valid', 'test'
            which determines the partition that scaffold belongs

        Returns
        -------
        bool
            A bool. True if the chromosome is sane.
        """
        ntasks = self.dataset.y.shape[1]
        train_ind, valid_ind, test_ind = self.split_chromosome_to_compound_split(split_chromosome)
        for task in range(ntasks):
            train_y = self.dataset.y[train_ind, task]
            train_y = train_y[~np.isnan(train_y)]

            valid_y = self.dataset.y[valid_ind, task]
            valid_y = valid_y[~np.isnan(valid_y)]

            test_y = self.dataset.y[test_ind, task]
            test_y = test_y[~np.isnan(test_y)]

            # one task has 0 data points
            if (len(train_y)==0) or (len(valid_y)==0) or (len(test_y)==0):
                return False

        return True

    def grade(self, split_chromosome: List[str]) -> float:
        """Assigns a score to a given chromosome

        Returns a total fitness score for a candidate split, as a weighted sum of
        fitness terms, with the weights determined by user specified split parameters.

        Args:
            split_chromosome (List[str]):
                A list of strings of length equal to the number of superscaffolds. The i'th string
                must be 'train', 'valid', or 'test', indicating which partition i'th scaffold is
                assigned to in the candidate split.

        Returns:
            float
                The total fitness score, ranging from 0 to 1 with 1 being the best.
        """
        
        fitness_scores = self.get_fitness_scores(split_chromosome)
        return fitness_scores['total_fitness']

    def get_fitness_scores(self, split_chromosome):
        """
        Calculate fitness scores for a given split chromosome based on various fitness terms.

        This method computes individual fitness scores for different criteria (e.g., scaffold 
        distance distribution, partition size ratios, response value distribution) and combines them into a 
        total fitness score. Each fitness term is weighted according to its respective weight 
        parameter.

        Args:
            split_chromosome (List[str]): A list of strings representing the data split to evaluate.

        Returns:
            dict: A dictionary containing individual fitness scores for each term and the 
                  normalized total fitness score. Keys include:
                  - 'test_scaf_dist': Fitness score representing the Tanimoto distance distribution between test set
                    compounds and their nearest neighbors in the training set.
                  - 'valid_scaf_dist': Fitness score representing the Tanimoto distance distribution between validation set
                    compounds and their nearest neighbors in the training set.
                  - 'ratio': Fitness score based on how well the proportions of compounds in the split subsets conform
                    to those specified by the split_valid_frac and split_test_frac parameters.
                  - 'response_distr': Fitness score based on how well the validation and test set response value 
                    distributions match that of the train subset.
                  - 'total_fitness': The normalized total fitness score, ranging from 0 to 1.
        """
        fitness_scores = {}
        total_fitness = 0.0
        # Only call the functions for each fitness term if their weight is nonzero
        if self.diff_fitness_weight_tvt != 0.0:
            #fitness_scores['test_scaf_dist'] = self.scaffold_diff_fitness(split_chromosome, 'train', 'test')
            fitness_scores['test_scaf_dist'] = self.far_frac_fitness(split_chromosome, 'train', 'test')
            total_fitness += self.diff_fitness_weight_tvt * fitness_scores['test_scaf_dist']
        if self.diff_fitness_weight_tvv != 0.0:
            #fitness_scores['valid_scaf_dist'] = self.scaffold_diff_fitness(split_chromosome, 'train', 'valid')
            fitness_scores['valid_scaf_dist'] = self.far_frac_fitness(split_chromosome, 'train', 'valid')
            total_fitness += self.diff_fitness_weight_tvv * fitness_scores['valid_scaf_dist']
        if self.ratio_fitness_weight != 0.0:
            fitness_scores['ratio'] = self.ratio_fitness(split_chromosome)
            total_fitness += self.ratio_fitness_weight * fitness_scores['ratio']
        if self.response_distr_fitness_weight != 0.0:
            fitness_scores['response_distr'] = self.response_distr_fitness(split_chromosome)
            total_fitness += self.response_distr_fitness_weight * fitness_scores['response_distr']
        # Normalize the score to the range [0,1]
        total_fitness /= (self.diff_fitness_weight_tvt + self.diff_fitness_weight_tvv + self.ratio_fitness_weight +
                    self.response_distr_fitness_weight)
        fitness_scores['total_fitness'] = total_fitness
        return fitness_scores

    def init_scaffolds(self,
             dataset: Dataset) -> None:
        """Creates super scaffolds used in splitting.

        This function combines the scaffolds found in the input dataset into a smaller set of super scaffolds
        according to the mtss_num_super_scaffolds parameter.

        Parameters
        ----------
        dataset: Dataset
            Deepchem Dataset. The parameter w must be created and ids must contain
            smiles.
        Returns
        -------
        None
            Only sets member variables self.ss and self.scaffold_hists
        """
        # First assign each of the samples to a scaffold bin
        # list of lists. one list per scaffold
        big_ss = self.generate_scaffolds(dataset)

        # using the same strategy as scaffold split, combine the scaffolds
        # together until you have roughly 100 scaffold sets
        self.ss = smush_small_scaffolds(big_ss, num_super_scaffolds=self.num_super_scaffolds)
        logger.info(f"Requested {self.num_super_scaffolds} super scaffolds, produced {len(self.ss)} from {len(big_ss)} original scaffolds")

        # rows is the number of scaffolds
        # columns is number of tasks
        self.scaffold_hists = _generate_scaffold_hists(self.ss, dataset.w)

    def split(self,
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            diff_fitness_weight_tvt: float = 0,
            diff_fitness_weight_tvv: float = 0,
            response_distr_fitness_weight: float = 0,
            ratio_fitness_weight: float = 1,
            num_super_scaffolds: int = 20,
            num_pop: int = 100,
            num_generations: int=30,
            dist_thresh: float = 0.3,
            print_timings: bool = False,
            early_stopping_generations = 25,
            log_every_n: int = 10) -> Tuple:
        """Creates a split for the given datset.

        This function splits the dataset into a list of super scaffolds, then
        assigns each super scaffold into one of three partitions. The scaffolds
        are assigned using a GeneticAlgorithm which tries to maximize a fitness
        function for the split. The fitness function computes a weighted sum
        of various criteria for the quality of the split, with the weights determined
        by user specified parameters; see the descriptions of the weight parameters
        below.

        Parameters
        ----------
        dataset: Dataset
            Deepchem Dataset. The parameter w must be created and ids must contain
            smiles.
        frac_train: float
            The fraction of data that each task should have in the train partition
        frac_valid: float
            The fraction of data that each task should have in the valid partition
        frac_test: float
            The fraction of data that each task should have in the test partition
        seed: Optional[int]
            Seed for random number generator
        diff_fitness_weight_tvt: float
            Weight for the importance of the difference between training and test
            partitions
        diff_fitness_weight_tvv: float
            Weight for the importance of the difference between training and valid
            partitions
        ratio_fitness_weight: float
            Weight for the importance of ensuring each task has the appropriate
            number of samples in training/validation/test
        response_distr_fitness_weight: float
            Weight for the importance of matching the response value distributions in
            the validation and test sets to that of the training set
        num_super_scaffolds: int
            The number of super scaffolds.
        num_pop: int
            Size of the population for the genetic algorithm
        num_generations: int
            Number of generations to run the genetic algorithm
        log_every_n: int
            Controls the logger by dictating how often logger outputs will be produced.
        Returns
        -------
        Tuple
            A tuple with 3 elements that are training, validation, and test compound
            indices into dataset, respectively
        """
        self.seed = seed

        self.dataset = dataset
        self.diff_fitness_weight_tvt = diff_fitness_weight_tvt
        self.diff_fitness_weight_tvv = diff_fitness_weight_tvv
        self.ratio_fitness_weight = ratio_fitness_weight
        self.response_distr_fitness_weight = response_distr_fitness_weight
        self.num_super_scaffolds = num_super_scaffolds
        self.num_pop = num_pop
        self.num_generations = num_generations
        self.dist_thresh = dist_thresh

        self.frac_train = frac_train
        self.frac_valid = frac_valid
        self.frac_test = frac_test

        # set up super scaffolds
        self.init_scaffolds(self.dataset)

        # ecpf features
        self.smiles = dataset.ids
        self.ecfp_features = calc_ecfp(dataset.ids)

        # calculate ScaffoldxScaffold distance matrix
        if (self.diff_fitness_weight_tvv > 0.0) or (self.diff_fitness_weight_tvt > 0.0):
            self._generate_scaffold_dist_matrix()

        # initial population
        population = []
        for i in range(self.num_pop):
            split_chromosome = self._split(frac_train=frac_train, frac_valid=frac_valid, 
                                frac_test=frac_test)

            population.append(split_chromosome)

        self.fitness_terms = {}
        gene_alg = ga.GeneticAlgorithm(population, self.grade, ga_crossover,
                        ga_mutate, self.seed)
        #gene_alg.iterate(num_generations)
        best_ever = None
        best_ever_fit = -np.inf
        best_gen = 0
        for i in range(self.num_generations):
            gene_alg.step(print_timings=print_timings)
            best_fitness = gene_alg.pop_scores[0]
            if best_fitness > best_ever_fit:
                best_ever = gene_alg.pop[0]
                best_ever_fit = best_fitness
                best_gen = i
            score_dict = self.get_fitness_scores(best_ever)
            for term, score in score_dict.items():
                self.fitness_terms.setdefault(term, []).append(score)
            if i % log_every_n == 0:
                logger.info(f"generation {i}: Best fitness {best_fitness:.3f}, best ever {best_ever_fit:.3f} at generation {best_gen}")
            if (best_fitness <= best_ever_fit) and (i - best_gen >= early_stopping_generations):
                logger.info(f"No fitness improvement after {early_stopping_generations} generations")
                break

        logger.info(f"Final best fitness score: {best_ever_fit:.3f} at generation {best_gen}")
        result = self.split_chromosome_to_compound_split(best_ever)
        return result

    def _split(self,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1) -> List[str]:
        """Return indices for specified split
        Parameters
        ----------
        frac_train: float, optional (default 0.8)
            The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
            The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
            The fraction of data to be used for the test split.
        Returns
        -------
        List[str]
            A list of strings, index i contains a string, 'train', 'valid', 'test'
            which determines the partition that scaffold belongs to
        """
        # Figure out how many positive samples we want for each task in each dataset.
        n_tasks = self.scaffold_hists.shape[1]
        y_present = self.scaffold_hists != 0
        indices_for_task = [
                # look at which scaffolds contain samples for a task
                # randomly shuffle the indices
                np.nonzero(y_present[:, i])[0]
                #np.random.permutation(np.nonzero(y_present[:, i])[0])
                for i in range(n_tasks)
        ]
        count_for_task = np.array([len(x) for x in indices_for_task])
        train_target = np.round(frac_train * count_for_task).astype(int)
        valid_target = np.round(frac_valid * count_for_task).astype(int)
        test_target = np.round(frac_test * count_for_task).astype(int)

        # Assign the positive samples to datasets.    Since a sample may be positive
        # on more than one task, we need to keep track of the effect of each added
        # sample on each task.    To try to keep everything balanced, we cycle through
        # tasks, assigning one positive sample for each one.
        train_counts = np.zeros(n_tasks, int)
        valid_counts = np.zeros(n_tasks, int)
        test_counts = np.zeros(n_tasks, int)
        set_target = [train_target, valid_target, test_target]
        set_counts = [train_counts, valid_counts, test_counts]
        set_inds: List[List[int]] = [[], [], []]
        assigned = set()
        for i in range(len(self.scaffold_hists)):
            for task in range(n_tasks):
                indices = indices_for_task[task]
                if i < len(indices) and indices[i] not in assigned:
                    # We have a sample that hasn't been assigned yet. Assign it to
                    # whichever set currently has the lowest fraction of its target for
                    # this task.

                    index = indices[i]
                    set_frac = [
                            1 if set_target[i][task] == 0 else
                            set_counts[i][task] / set_target[i][task] for i in range(3)
                    ]
                    s = np.argmin(set_frac)
                    set_inds[s].append(index)
                    assigned.add(index)
                    set_counts[s] += self.scaffold_hists[index]

        split_chromosome = ['']*len(self.ss)
        for part_name, scaffolds in zip(['train', 'valid', 'test'], set_inds):
            for s in scaffolds:
                split_chromosome[s] = part_name

        return split_chromosome

    def train_valid_test_split(self, 
            dataset: Dataset,
            frac_train: float = 0.8,
            frac_valid: float = 0.1,
            frac_test: float = 0.1,
            seed: Optional[int] = None,
            diff_fitness_weight_tvt: float = 0,
            diff_fitness_weight_tvv: float = 0,
            ratio_fitness_weight: float = 1,
            response_distr_fitness_weight: float = 0,
            num_super_scaffolds: int = 20,
            num_pop: int = 100,
            num_generations: int=30,
            train_dir: Optional[str] = None,
            valid_dir: Optional[str] = None,
            test_dir: Optional[str] = None,
            dist_thresh: float = 0.3,
            log_every_n: int = 10) -> Tuple[Dataset, Dataset, Dataset]:
        """Creates a split for the given datset

        This split splits the dataset into a list of super scaffolds then
        assigns each scaffold into one of three partitions. The scaffolds
        are assigned using a GeneticAlgorithm and tries to maximize the
        difference between the training and test partitions as well as ensuring
        all tasks have an appropriate number of training/validation/test samples

        Parameters
        ----------
        dataset: Dataset
            Deepchem Dataset. The parameter w must be created and ids must contain
            smiles.
        frac_train: float
            The fraction of data that each task should have in the train partition
        frac_valid: float
            The fraction of data that each task should have in the valid partition
        frac_test: float
            The fraction of data that each task should have in the test partition
        seed: Optional[int]
            Seed for random number generator
        diff_fitness_weight_tvt: float
            Weight for the importance of the difference between training and test
            partitions
        diff_fitness_weight_tvv: float
            Weight for the importance of the difference between training and valid
            partitions
        ratio_fitness_weight: float
            Weight for the importance of ensuring each task has the appropriate
            number of samples in training/validation/test
        num_super_scaffolds: int
            The number of super scaffolds.
        num_pop: int
            Size of the population for the genetic algorithm
        log_every_n: int
            Controls the logger by dictating how often logger outputs will be produced.
        num_generations: int
            Number of generations to run the genetic algorithm
        train_dir: str, optional (default None)
            If specified, the directory in which the generated
            training dataset should be stored. This is only
            considered if `isinstance(dataset, dc.data.Dataset)`
        valid_dir: str, optional (default None)
            If specified, the directory in which the generated
            valid dataset should be stored. This is only
            considered if `isinstance(dataset, dc.data.Dataset)`
            is True.
        test_dir: str, optional (default None)
            If specified, the directory in which the generated
            test dataset should be stored. This is only
            considered if `isinstance(dataset, dc.data.Dataset)`
            is True.

        Returns
        -------
        Tuple
            A tuple with 3 elements that are training, validation, and test compound
            indices into dataset, respectively
        """
        train_inds, valid_inds, test_inds = self.split(
            dataset=dataset, frac_train=frac_train,
            frac_valid=frac_valid, frac_test=frac_test, 
            seed=seed, diff_fitness_weight_tvt=diff_fitness_weight_tvt,
            diff_fitness_weight_tvv=diff_fitness_weight_tvv, ratio_fitness_weight=ratio_fitness_weight,
            response_distr_fitness_weight=response_distr_fitness_weight,
            num_super_scaffolds=num_super_scaffolds, num_pop=num_pop, num_generations=num_generations,
            dist_thresh=0.3,
            print_timings=False)

        if train_dir is None:
            train_dir = tempfile.mkdtemp()
        if valid_dir is None:
            valid_dir = tempfile.mkdtemp()
        if test_dir is None:
            test_dir = tempfile.mkdtemp()

        train_dataset = dataset.select(train_inds, train_dir)
        valid_dataset = dataset.select(valid_inds, valid_dir)
        test_dataset = dataset.select(test_inds, test_dir)
        if isinstance(train_dataset, Dataset):
            train_dataset.memory_cache_size = 40 * (1 << 20)  # 40 MB

        return train_dataset, valid_dataset, test_dataset

def ga_crossover(parents: List[List[str]],
                num_pop: int,
                random_state: np.random.Generator) -> List[List[str]]:
    """Create the next generation from parents

    A random index is chosen and genes up to that index from
    the first chromosome is used and genes from the index to
    the end is used.

    Parameters
    ----------
    parents: List[List[str]]
        A list of chromosomes.
    num_pop: int
        The number of new chromosomes to make
    random_state: np.random.Generator
        Random number generator
    Returns
    -------
    List[List[str]]
        A list of chromosomes. The next generation
    """
    # just single crossover point
    new_pop = []
    for i in range(num_pop):
        parent1 = parents[i%len(parents)]
        parent2 = parents[(i+1)%len(parents)]

        crossover_point = random_state.integers(low=0, high=len(parents[0])-1, size=1)[0]
        new_pop.append(parent1[:crossover_point]+parent2[crossover_point:])

    return new_pop

def ga_mutate(new_pop: List[List[str]],
            random_state: np.random.Generator,
            mutation_rate: float = .02,) -> List[List[str]]:
    """Mutate the population

    Each chromosome is copied and mutated at mutation_rate.
    When a gene mutates, it's randomly assigned to a partiton.
    possibly the same partition.

    Parameters
    ----------
    new_pop: List[List[str]]
        A list of chromosomes.
    random_state: np.random.Generator
        Random number generator
    mutation_rate: float
        How often a mutation occurs. 0.02 is a good rate for
        my test sets.
    Returns
    -------
    List[List[str]]
        A list of chromosomes. Mutated chromosomes.
    """
    mutated = []
    for solution in new_pop:
        new_solution = list(solution)
        for i, gene in enumerate(new_solution):
            if random_state.random() < mutation_rate:
                new_solution[i] = ['train', 'valid', 'test'][random_state.integers(low=0, high=2, size=1)[0]]
        mutated.append(new_solution)

    return mutated

def make_y_w(dataframe: pd.DataFrame, 
            columns: List[str]) -> Tuple:
    """Create y and w matrices for Deepchem's Dataset

    Extracts labels and builds the w matrix for a dataset.
    The w matrix contains a 1 if there's a label and 0
    if not.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Pandas DataFrame
    columns: List[str]
        A list of columns that contain labels.
    Returns
    -------
    Tuple
        Two numpy arrays, y and w.
    """
    y = dataframe[columns].values
    w = np.ones_like(y)
    nan_indx = np.argwhere(np.isnan(y))
    for r, c in nan_indx:
        w[r, c] = 0

    return y, w

def split_using_MultitaskScaffoldSplit(df: pd.DataFrame,
                    id_col: str,
                    target_cols: List[str],
                    smiles_col: str,
                    **kwargs) -> pd.DataFrame:
    """Produces an AMPL compatible split file given a dataframe

    Parameters
    ----------
    df: pd.Dataframe
        Dataframe containing compounds to split
    id_col: str
        Column containing compound ids
    target_cols: List[str]
        List of target columns. Can be of length 1
    smiles_col: str
        Column containing base_rdkit_smiles strings
    **kwargs:
        Any arguments you want to pass to MultitaskScaffoldSplit

    Returns
    -------
    pd.DataFrame
        Returns a DataFrame that's compatible with AMPL. Plus
        an extra column that shows which scaffold each compound was
        assigned to.
    """

    # Build a deepchem Dataset. X isn't used and can be ignored
    X = np.ones((len(df), 10))
    y, w = make_y_w(df, target_cols)
    ids = df[smiles_col].values

    # build deepchem Dataset
    dataset = dc.data.NumpyDataset(X, y, w=w, ids=ids)
    mss = MultitaskScaffoldSplitter()
    splits = mss.split(dataset, **kwargs)

    split_df = pd.DataFrame({'cmpd_id':df[id_col].values,
                            'fold': [0]*len(df)})
    split_df['subset'] = ['']*split_df.shape[0]
    split_df['subset'].iloc[splits[0]] = 'train'
    split_df['subset'].iloc[splits[1]] = 'valid'
    split_df['subset'].iloc[splits[2]] = 'test'

    return split_df

def split_with(df, splitter, smiles_col, id_col, response_cols, **kwargs):
    """Given a dataframe and a splitter, perform split
    Return a split dataframe, with base_rdkit_smiles as key
    and subset with train, valid, test
    """
    # Build a deepchem Dataset. X isn't used and can be ignored
    X = np.ones((len(df), 10))
    y, w = make_y_w(df, response_cols)
    ids = df[smiles_col].values

    dataset = dc.data.NumpyDataset(X, y, w=w, ids=ids)

    splits = splitter.split(dataset, **kwargs)

    split_df = pd.DataFrame(df[[id_col]])
    split_df = split_df.rename(columns={id_col:'cmpd_id'})
    split_array = np.array(['unassigned']*split_df.shape[0])
    split_array[splits[0]] = 'train'
    split_array[splits[1]] = 'valid'
    split_array[splits[2]] = 'test'

    split_df['subset'] = split_array

    if 'ss' in dir(splitter):
        ss = splitter.ss
        scaffold_array = np.ones(split_df.shape[0])
        for i, scaffold in enumerate(ss):
            scaffold_array[list(scaffold)] = i

        split_df['scaffold'] = scaffold_array

    return split_df

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', type=str, help='path to input csv')
    parser.add_argument('dist_weight', type=float, 
        help='Weight for the importance of the difference between training and test partitions')
    parser.add_argument('ratio_weight', type=float, 
        help='Weight for the importance of ensuring each task has the appropriate number of samples in training/validation/test')
    parser.add_argument('num_gens', type=int, 
        help='Number of generations to run.')
    parser.add_argument('smiles_col', type=str, help='the column containing smiles')
    parser.add_argument('id_col', type=str, help='the column containing ids')
    parser.add_argument('response_cols', type=str, help='comma seperated string of response columns')
    parser.add_argument('output', type=str, help='name of the split file')
    parser.add_argument('seed', type=int, default=0, help='Random seed used in random number generators.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    total_df = pd.read_csv(args.data)

    dfw = args.dist_weight
    rfw = args.ratio_weight

    response_cols = args.response_cols.split(',')

    mss = MultitaskScaffoldSplitter()
    mss_split_df = split_with(total_df, mss, 
        smiles_col=args.smiles_col, id_col=args.id_col, response_cols=response_cols, 
        diff_fitness_weight=dfw, ratio_fitness_weight=rfw, num_generations=args.num_gens,
        seed=args.seed)
    mss_split_df.to_csv(args.output, index=False)
