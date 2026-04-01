import numpy as np
import uuid
import scipy.spatial.distance as scipy_distance
import multiprocessing
from tqdm import tqdm
import timeit
from typing import Any, Callable, List, Tuple, Optional

N_PROCS = multiprocessing.cpu_count()

class GeneticAlgorithm:
    """ A Genetic algorithm for finding the best split for a dataset

    This class implements a basic genetic function. It handles the basic outline
    and takes as input functions that every genetic algorithm would need.

    The population is always kept in a sorted state where the highest scoring
    chromosome is always the first in the list.
    """

    def __init__(self,
                init_pop: List[List[Any]],
                fitness_func: Callable,
                crossover_func: Callable,
                mutate_func: Callable,
                seed: Optional[int]):
        """
        Creates a GeneticAlgorithm object

        Parameters
        ----------
        init_pop: List[List[Any]]
            A population is a list of chromosomes and a chromosome is a list of objects
        fitness_func: Callable
            A callable that takes a chromosome as input and returns a floating
            point value. higher the better
        crossover_func: Callable
            A callable that takes the parents, and the number of children
            desired in the next generation. Returns a list of chromosomes representing
            the next generation
        mutate_func: Callable
            A callable that takes a list of chromosomes and returns another list of mutated 
            chromosomes
        seed: Optional[int]
            Seed for random number generator
        """

        if seed is None:
            seed = uuid.uuid4().int % (2**32)
        self.random_state = np.random.default_rng(seed)

        self.pop = init_pop
        self.pop_scores = None
        self.num_pop = len(init_pop)
        self.num_parents = int(len(self.pop)/2)
        self.fitness_func = fitness_func
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.serial_grade_population()
        #self.parallel_grade_population()


    def serial_grade_population(self):
        """ Scores the chromosomes in the current population and sorts them in decreasing score order.
        Saves the sorted scores in self.pop_scores. Not multithreaded; surprisingly, this runs faster
        than the multithreaded function parallel_grade_scores.

        Parameters
        ----------
        None

        Returns
        -------
        None. As a side effect, sorts the chromosomes in self.pop and updates the scores in self.pop_scores.
        """
        fitnesses = []
        for chrom in self.pop:
            fitnesses.append(self.fitness_func(chrom))
        pairs = sorted(zip(fitnesses, self.pop), reverse=True)
        self.pop = [chrome for fitness, chrome in pairs]
        self.pop_scores = [fitness for fitness, chrome in pairs]


        self.best_fitness = -1e20
        self.best_solution = None

    def update_best_solution(self):
        """ Grade the population and save the scores

        Saves the best solution from self.pop and self.pop_scores

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.pop_scores[0]>self.best_fitness:
            self.best_fitness = self.pop_scores[0]
            self.best_solution = self.pop[0]

    def get_best(self) -> Tuple[float,List[Any]]:
        """ Grade the population and save the scores

        Saves the best solution from self.pop and self.pop_scores

        Parameters
        ----------
        None

        Returns
        -------
        best_fitness: float
            The best fitness ever scored from any generation.

        best_solution: List[Any]
            The best chromosome from any generation
        """

        return self.best_fitness, self.best_solution

    def parallel_grade_population(self):
        """ Scores the chromosomes in the current population and sorts them in decreasing score order.
        Saves the sorted scores in self.pop_scores. 
        
        Although this does the same thing in multiple threads as the single-threaded function 
        serial_grade_population, it seems to run much slower, at least for multitask scaffold splits
        with 100 chromosomes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pool = multiprocessing.Pool(processes=N_PROCS)
        fitnesses = pool.map(self.fitness_func, self.pop)
        pool.close()
        pool.join()
        pairs = sorted(zip(fitnesses, self.pop), reverse=True)
        self.pop = [chrome for fitness, chrome in pairs]
        self.pop_scores = [fitness for fitness, chrome in pairs]

    def select_parents(self) -> List[List[Any]]:
        """ Looks at self.pop and chooses parents for the next generation

        The method used here uses the top scoring self.num_parents chromosomes
        as the parents of the next generation

        Parameters
        ----------
        None

        Returns
        -------
        parents: List[List[Any]]
            A list of chromosomes that will be parents for the next generation
        """
        return self.pop[:self.num_parents]

    def iterate(self, num_generations: int):
        """ Iterates the genetic algorithm num_generations

        Calling self.step once iterates one generation. 
        This function iteraties num_generations times.

        Parameters
        ----------
        num_generations: int
            The number of generations you'd like to simulate

        Returns
        -------
        Nothing
        """
        for i in tqdm(range(num_generations)):
            self.step()

    def step(self, print_timings: bool = False):
        """Simulates one generation

        Takes one step in the generation

        Parameters
        ----------
        print_timings : bool
            Boolean that turns on/off print timings

        Returns
        -------
        Nothing
        """

        start = timeit.default_timer()
        i = timeit.default_timer()
        # select parents using rank selection
        parents = self.select_parents()
        if print_timings:
            print('\tfind parents %0.2f min'%((timeit.default_timer()-i)/60))

        i = timeit.default_timer()
        # Generate new population by crossing parent chromosomes
        new_pop = self.crossover_func(parents, self.num_pop, random_state=self.random_state)
        if print_timings:
            print('\tcrossover %0.2f min'%((timeit.default_timer()-i)/60))

        # mutate population
        i = timeit.default_timer()
        self.pop = self.mutate_func(new_pop, random_state=self.random_state)
        # Compute scores for new chromosomes and sort population by score
        self.serial_grade_population()
        #self.parallel_grade_population()
        if print_timings:
            print('\tmutate %0.2f min'%((timeit.default_timer()-i)/60))
            print('total %0.2f min'%((timeit.default_timer()-start)/60))

        self.update_best_solution()

if __name__ == '__main__':
    num_pop = 500
    num_genes = 40
    init_pop = [list(np.random.binomial(1, .5, size=num_genes)) for i in range(num_pop)]

    target_chromosome = np.random.binomial(1, .3, size=num_genes)
    def fitness_func(chromosome):
        return 1 - scipy_distance.rogerstanimoto(chromosome, target_chromosome)

    def crossover_func(parents, pop_size, random_state):
        new_pop = []
        for i in range(num_pop):
            parent1 = parents[i%len(parents)]
            parent2 = parents[(i+1)%len(parents)]

            crossover_point = random_state.integers(0, len(parents[0])-1, 1)[0]
            new_pop.append(parent1[:crossover_point]+parent2[crossover_point:])

        return new_pop

    def mutate_func(pop, random_state, mutate_chance=0.01):
        new_pop = []
        for chromosome in pop:
            new_chromosome = list(chromosome)
            for i, g in enumerate(new_chromosome):
                if random_state.random() < mutate_chance:
                    if new_chromosome[i] == 0:
                        new_chromosome[i] = 1
                    else:
                        new_chromosome[i] = 0
            new_pop.append(new_chromosome)

        return new_pop

    ga = GeneticAlgorithm(init_pop, fitness_func, crossover_func, mutate_func)
    print('best scores')
    for i in range(50):
        ga.step()
        if i % 10 == 0:
            print('%0.2f'% ga.pop_scores[0])

    print('target:  ',target_chromosome)
    best_fit = ga.pop_scores[0]
    best = np.array(ga.pop[0])
    print('closets: ', best.astype(int))
    print('best fitness %0.2f'%best_fit)
