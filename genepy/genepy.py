#!/usr/bin/env python

import sys
import numpy as np
from tqdm import tqdm

def bin_str2arr(bin_str):
    """
    Convert a binary string (chromosome) to a numpy array of 0s and 1s.
    
    Parameters:
    -----------
    bin_s : str
    A string representing a binary number, e.g., '101011'.
        
    Returns:
    --------
    np.ndarray
    A numpy array of 0s and 1s corresponding to the input binary string.
    """
    l	= list(bin_str)
    return np.array(l, dtype = 'int')


def bin_arr2str(bin_arr):
    """
    Convert a numpy array of bits to a binary string (chromosome).

    Parameters:
    -----------
    bin_arr : np.ndarray
    A numpy array of 0s and 1s corresponding to the chromosome.

    Returns:
    --------
    bin_s : str
    The string representing the input binary vector, e.g., '101011'.
    """
    bin_str = ''.join(list(bin_arr.astype(int).astype(str)))
    return bin_str


def dec2bin(dec_arr, N_bits_chromosome, decimal_acc, offset = 0):
    """
    Convert a vector of decimal numbers to a concatenated binary string (chromosome representation).
    
    Parameters:
    -----------
    dec_arr : np.ndarray
    Input decimal vector
        
    N_bits_chromosome : int
    Desired size of the full chromosomes in bits
        
    decimal_acc : int
    Desired decimal accuracy (number of decimal places)
    
    offset : float [np.ndarray], optional
    Any offset (vector) to be added to the input decimal array before conversion. Defaults to 0.
        
    Returns:
    --------
    str
    The chromosome representation of the input decimal vector
    """
    m 	    = []
    N_var   = len(dec_arr)
    divsize = N_bits_chromosome // N_var
    dec_arr = dec_arr + offset
    for i in range(N_var):
        num 	= int(round(dec_arr[i], decimal_acc)*(10**decimal_acc))
        num_bin = format(num, 'b')
        if len(num_bin) < divsize: # zero padding
            num_bin = ('0'*(divsize - len(num_bin))) + num_bin
        m.append(num_bin)
    bin_str = "".join(m)
    return bin_str


def bin2dec(bin_arr, N_bits_segment, decimal_acc, offset = 0):
    """
    Convert a chromosome vector of bits (0s and 1s) to a decimal vector (phenotype representation).
    
    Parameters:
    -----------
    bin_arr : np.ndarray
    Input chromosome vector (could have been created using `bin_str2arr`)
        
    N_bits_segment : int
    Size of each independent variable of the vector in bits (also called a segment of the chromosome)
        
    decimal_acc : int
    Desired decimal accuracy (number of decimal places)
    
    offset : float [np.ndarray], optional
    Any offset (vector) to be subtracted from the decimal array after conversion. Defaults to 0.
        
    Returns:
    --------
    np.ndarray
    The phenotype representation (decimal vector) of the input chromosome (binary vector)
    """
    N 		= len(bin_arr)
    N_var 	= N // N_bits_segment
    dec_arr = []
    for i in range(N_var):
        var = bin_arr2str(bin_arr)
        var = int(var, 2)
        var = var / 10**(decimal_acc)
        dec_arr.append(var)
    dec_arr = np.array(dec_arr) - offset
    return dec_arr


def pop_dec2bin(pop_dec, N_bits_chromosome, decimal_acc, offset = 0):
    """
    Convert a population of decimal vectors to their genotype representation (binary vectors).

    Parameters:
    -----------
    pop_dec : np.ndarray
    A population in phenotype representation (array of shape `(popsize, N_var)` containing decimal vectors of individuals)

    N_bits_chromosome : int
    Desired size of a full chromosome in bits

    decimal_acc : int
    Desired decimal accuracy (number of decimal places)

    offset : float [np.ndarray], optional
    Any offset (vector) to be added to the input decimal arrays before conversion. Defaults to 0.

    Returns:
    --------
    str
    The genotype representation (binary vectors) of the input decimal population
    """
    pop_bin = []
    for indiv in pop_dec:
        indiv_bin = dec2bin(
            dec_arr = indiv,
            N_bits_chromosome = N_bits_chromosome,
            decimal_acc = decimal_acc,
            offset = offset,
        )
        pop_bin.append(bin_str2arr(indiv_bin))
    return np.array(pop_bin)


def pop_bin2dec(pop_bin, N_bits_segment, decimal_acc, offset = 0):
    """
    Convert a population of binary vectors to their phenotype representation (decimal vectors).

    Parameters:
    -----------
    pop_bin : np.ndarray
    A population in genotype representation (array of shape `(popsize, N_bits_chromosome)` containing binary vectors of individuals)

    N_bits_segment : int
    Size of each independent variable (of an individual vector in the population) in bits (also called a segment of a chromosome)

    decimal_acc : int
    Desired decimal accuracy (number of decimal places)

    offset : float [np.ndarray], optional
    Any offset (vector) to be subtracted from the decimal arrays after conversion. Defaults to 0.

    Returns:
    --------
    np.ndarray
    The phenotype representation (decimal vector) of the input binary population
    """
    pop_dec = []
    for indiv in pop_bin:
        indiv_dec = bin2dec(
            bin_arr = indiv,
            N_bits_segment = N_bits_segment,
            decimal_acc = decimal_acc,
            offset = offset,
        )
        pop_dec.append(indiv_dec)
    return np.array(pop_dec)


def crossover(
    mating_pool, 
    type = 'uniform',
    seed = 42,
    random_rng = None,
):
    """
    Perform crossover on the mating pool to produce offspring.

    Parameters:
    -----------
    mating_pool : np.ndarray
    A mating pool (pairs of individuals) of shape `(N_pairs, 2, N_bits_chromosome)`.

    type : str, optional
    Type of (numerical) crossover to perform. Currently only 'uniform' is implemented. Defaults to 'uniform'.

    seed : int, optional
    Pseudorandom seed for reproducibility of crossover. Defaults to 42. Overridden by `random_rng` if supplied.

    random_rng : np.random.Generator, optional
    Supplied if an instance of the random number generator exists that must be used for creating pseudorandom numbers for crossover. Overrides `seed`. Defaults to `None`.
    """
    offspring = []
    if random_rng is None:
        random_rng   = np.random.default_rng(seed)
    if type == 'uniform':
        for i in range(len(mating_pool)):
            g1, g2 = mating_pool[i]
            for j in range(len(g1)):
                r = random_rng.random()
                if r > 0.5:
                    tmp   = g1[j]
                    g1[j] = g2[j]
                    g2[j] = tmp
            offspring.append(g1)
            offspring.append(g2)
        offspring = np.array(offspring)
        return offspring
    else:
        raise ValueError("Crossover type not recognized. Currently only 'uniform' works.")
    

def mutate(
    pop, 
    prob_mut,
    seed = 42,
    random_rng = None,
):
    """
    Mutate a population by flipping bits with a given mutation probability.

    Parameters:
    -----------
    pop : np.ndarray
    A population in genotype representation (array of shape `(popsize, N_bits_chromosome)` containing binary vectors of individuals)

    prob_mut : float
    Probability of mutation for each bit (between 0 and 1)

    seed : int, optional
    Pseudorandom seed for reproducibility of mutation. Defaults to 42. Overridden by `random_rng` if supplied.

    random_rng : np.random.Generator, optional
    Supplied if an instance of the random number generator exists that must be used for creating pseudorandom numbers for mutation. Overrides `seed`. Defaults to `None`.
    """
    if prob_mut == 0.0:
        return pop
    elif prob_mut < 0.0 or prob_mut > 1.0:
        raise ValueError("Mutation probability must be between 0 and 1.")
    else:
        if random_rng is None:
            random_rng = np.random.default_rng(seed)
        for i in range(len(pop)):
            for j in range(len(pop[i])):
                r = random_rng.random()
                if r < prob_mut:
                    pop[i][j] = float(not(pop[i][j]))
        return pop


class PopGenetics:
    """ 
    A class encapsulating population genetics operations for a genetic algorithm, including population initialization, mating pool creation, breeding, and evolution over generations.
    """
    def __init__(
        self, 
        fitness_func, 
        N_var, 
        decimal_acc, 
        N_bits_chromosome,
        seed_popinit   = 42,
        seed_selection = 43,
        seed_crossover = 44,
        seed_mutation  = 45,
    ):
        """
        Initialize the population genetics class

        Parameters:
        -----------
        fitness_func : callable
        Function to evaluate the fitness of an individual or the entire population. Takes phenotype(s) as input.

        N_var : int
        Number of varibles in an individual vector -- the dimensionality of the problem to be solved

        decimal_acc : int
        Desired decimal accuracy (in decimal places)

        N_bits_chromosome : int
        Desired size of a full chromosome in bits

        seed_[popinit, selection, crossover, mutation] : int, optional
        Pseudorandom seed to initialize random number generators for creating the initial population, selecting pairs in mating pools, crossover, and mutation, respectively. Defaults to [42, 43, 44, 45].
        """
        self.fitness_func  = fitness_func
        self.N_var         = N_var
        self.decimal_acc   = decimal_acc
        self.N_bits_chromosome  = N_bits_chromosome
        self.N_bits_segment     = N_bits_chromosome // N_var
        self.rng_popinit   = np.random.default_rng(seed_popinit)
        self.rng_selection = np.random.default_rng(seed_selection)
        self.rng_crossover = np.random.default_rng(seed_crossover)
        self.rng_mutation  = np.random.default_rng(seed_mutation)
        
    def initialize_population(
        self, 
        popsize, 
        var_ranges,
        dist = 'uniform', 
        return_genotype = False,
        offset = 0,
    ):
        """ 
        Initialize a population by stochastically sampling individuals in the required variable ranges

        Parameters:
        -----------
        popsize : int
        Desired number of individuals in the initial population (population size)

        var_ranges : np.ndarray
        Lower and upper boundaries (inclusive) of variables to sample the initial population within, array of shape `(2, N_var)`.

        dist : str, optional
        Prior probability distribution for sampling the initial population. Currently only 'uniform' is supported. Defaults to 'uniform'.

        return_genotype : bool, optional
        Whether to return the initial population in the genotype representation (binary vectors). Defaults to False.

        offset : float [np.ndarray], optional
        Any offset (vector) to be added to the decimal arrays before conversion to the binary genotype. Defaults to 0.

        Returns:
        -----------
        np.ndarray
        Initial population of shape `(popsize, N_var)` or `(popsize, N_bits_chromosome)` (if `return_genotype` is `True`)
        """
        self.offset = offset
        pop = []
        if dist == 'uniform':
            pop = self.rng_popinit.uniform(
                low  = var_ranges[0],
                high = var_ranges[1],
                size = (popsize, self.N_var)
            )
        else:
            raise ValueError("Distribution type not recognized. Currently only 'uniform' works.")
        if return_genotype:
            pop_bin = []
            for indiv in pop:
                indiv_bin = dec2bin(
                    dec_arr = indiv,
                    N_bits_chromosome = self.N_bits_chromosome,
                    decimal_acc = self.decimal_acc,
                    offset = offset,
                )
                pop_bin.append(
                    bin_str2arr(indiv_bin)
                )
            pop_bin = np.array(pop_bin)
            return pop_bin
        else:
            return pop
        

    def create_mating_pool(
        self,
        pop, 
        fitness_arr,
        N_pairs,
        selection_type = 'SUS', 
        rank_selection = False,
    ):
        """
        Create a mating pool by selecting pairs of individuals from the given population based on their fitness (selection pressure).

        Parameters:
        -----------
        pop : np.ndarray
        Binary population array of shape `(popsize, N_bits_chromosome)` where `N_bits_chromosome` is the size of each chromosome in bits. 

        fitness_arr : np.ndarray
        1D array of fitness values corresponding to the individuals in the (breeding) population.

        N_pairs : int
        Number of pairs to be selected for mating. (Pairs are selected with replacement - one pair can produce multiple independent children.)

        selection_type : str, optional
        Type of parent selection method to use. Options are: 'RW' for Roulette Wheel selection and 'SUS' for Stochastic Universal Sampling. The latter is a variation of the former with N_dim (angularly) equidistant pointers on the roulette wheel, where N_dim is the dimensionality of a seletion (e.g., for a "pair", N_dim = 2). Defaults to 'SUS'.

        rank_selection : bool, optional
        If `True`, applies a rank-based selection pressure instead of a fitness-proportionate one. Defaults to `False`.

        Returns:
        --------
        np.ndarray
        An array of shape `(N_pairs, 2, N_bits_chromosome)` - the mating pool.

        Raises:
        -------
        ValueError
        If an unrecognized selection type is requested. Currently recognized types are 'RW' and 'SUS'.
        """
        if fitness_arr is None:
            pop_dec = pop_bin2dec(
                pop_bin = pop,
                N_bits_segment = self.N_bits_segment,
                decimal_acc = self.decimal_acc,
                offset = self.offset
            )
            fitness_arr = self.fitness_func(pop_dec)

        # first sort the population according to fitness
        indsort 	 = np.argsort(fitness_arr)
        pop 		 = pop[indsort]
        fitness_arr  = fitness_arr[indsort]
        if rank_selection:
            ranks      = np.arange(len(pop), 0, -1)
            probs      = 1./ ranks
            probs      = probs / np.sum(probs) # normalization
            cuml_probs = np.array([sum(probs[0:k+1]) for k in range(0, len(pop))])
        else:
            probs      = fitness_arr / np.sum(fitness_arr) # normalization
            cuml_probs = np.array([sum(probs[0:k+1]) for k in range(0, len(pop))])
        
        pairs = []

        if selection_type.lower() == 'rw': 	# this is the simple roulette wheel selection type (for selecting individuals)
            for i in range(N_pairs):
                r1 	= self.rng_selection.random()
                r2 	= self.rng_selection.random()
                j 	= 0 
                k 	= 0
                while cuml_probs[j] < r1:
                    j 	= j + 1
                while cuml_probs[k] < r2:
                    k 	= k + 1
                if j == k:
                    if k == 0:
                        k = k + 1
                    else:
                        k = k - 1
                pair 	= [j, k]
                pairs.append(pop[pair])
        
        elif selection_type.lower() == 'sus': 	# this is the stochastic universal sampling
            for i in range(N_pairs):
                r1 	= self.rng_selection.random()*0.5
                r2 	= r1 + 0.5
                j 	= 0 
                k 	= 0
                while cuml_probs[j] < r1:
                    j 	= j + 1
                while cuml_probs[k] < r2:
                    k 	= k + 1
                if j == k:
                    if k == 0:
                        k = k + 1
                    else:
                        k = k - 1 
                pair 	= [j, k]
                pairs.append(pop[pair])

        else:
            raise ValueError("Selection type not recognized. Use 'RW' or 'SUS'.")
            
        return np.array(pairs)
    
    def breed(
        self,
        mating_pool,
        crossover_type = 'uniform',
        prob_mut = 0.0,
        prune = False,
        pruning_cutoff = None,
        return_fitness = False,
    ):
        """
        Breed the mating pool (pairs) by performing crossover on the chromosomes to produce offspring and then mutating them. An optional elitist pruning of the offspring based on their fitness values can be applied.

        Parameters:
        -----------
        mating_pool: np.ndarray
        The mating pool created using `create_mating_pool`, of shape `(N_pairs, 2, N_bits_chromosome)`.

        crossover_type : str, optional
        Type of (numerical) crossover to perform. Currently only 'uniform' is supported. Defaults to 'uniform'.

        prob_mut : float
        Probability of mutation for each bit (between 0 and 1)

        prune : bool, optional
        Whether to apply an "elitist" pruning of the offspring based on their fitness values. If `True`, a pruning threshold must be supplied. Defaults to `False`.

        pruning_cutoff : int, optional
        If pruning, the number of fittest offspring to keep. Must be supplied if `prune` is `True`. Defaults to `None`.

        return_fitness : bool, optional
        Whether to return the fitness values of the offspring along with the offspring. Defaults to `False`.

        Returns:
        --------
        np.ndarray or (np.ndarray, np.ndarray)
        The offspring produced after crossover, mutation and any pruning if applied. Returns the fitness values of the offspring as well (second returned object) if `return_fitness` is `True`.

        Raises:
        -------
        ValueError
        If an unrecognized crossover type is supplied. Currently only 'uniform' is recognized.

        ValueError
        If pruning is requested but a pruning cutoff is not supplied.
        """
        offspring = []
        offspring = crossover(
            mating_pool = mating_pool,
            type = crossover_type,
            random_rng = self.rng_crossover,
        )
        offspring = mutate(
            pop = offspring,
            prob_mut = prob_mut,
            random_rng = self.rng_mutation,
        )
        if return_fitness:
            offspring_dec = pop_bin2dec(
                pop_bin = offspring,
                N_bits_segment = self.N_bits_segment,
                decimal_acc = self.decimal_acc,
                offset = self.offset,
            )
            fitness_offspring = self.fitness_func(offspring_dec)
        if prune:
            if pruning_cutoff is None:
                raise ValueError("If pruning, a pruning cutoff (in number of offspring to keep) must be supplied.")
            if not return_fitness:
                offspring_dec = pop_bin2dec(
                    pop_bin = offspring,
                    N_bits_segment = self.N_bits_segment,
                    decimal_acc = self.decimal_acc,
                    offset = self.offset,
                )
                fitness_offspring = self.fitness_func(offspring_dec)
            indsort   = np.argsort(fitness_offspring)
            offspring = offspring[indsort][-pruning_cutoff:]
            fitness_offspring = fitness_offspring[indsort][-pruning_cutoff:]

        if return_fitness:
            return offspring, fitness_offspring
        else:
            return offspring
        
    def evolve(
        self, 
        pop, 
        N_gen, 
        N_pairs,
        selection_type = 'SUS',
        switch = None,
        crossover_type = 'uniform',
        prob_mut = 0.0,
        prune = False,
        pruning_cutoff = None,
        verbose = True,
    ):
        """
        Evolve a population of individuals by natural selection for a fixed number of generations.

        Parameters:
        -----------
        pop : np.ndarray 
        Binary population array of shape `(popsize, N_bits_chromosome)` where `N_bits_chromosome` is the size of each chromosome in bits. 

        N_gen : int
        Number of generation to evolve the population for 

        N_pairs: int 
        Number of mating pairs to pick in each mating pool 

        selection_type : str, optional
        Type of parent selection method to use. Options are: 'RW' for Roulette Wheel selection and 'SUS' for Stochastic Universal Sampling. The latter is a variation of the former with N_dim (angularly) equidistant pointers on the roulette wheel, where N_dim is the dimensionality of a seletion (e.g., for a "pair", N_dim = 2). Defaults to 'SUS'. 

        switch : int, optional 
        Optional switching generation for the selection type from fitness-proportionate to rank-based. Defaults to None (in which case no switching is applied).

        crossover_type : str, optional 
        Type of (numerical) crossover to perform. Currently only 'uniform' is supported. Defaults to 'uniform'. 

        prob_mut : float, optional 
        Probability of mutation for each bit (between 0 and 1)

        prune : bool, optional
        Whether to apply an "elitist" pruning of the offspring based on their fitness values. If `True`, a pruning threshold must be supplied. Defaults to `False`.

        pruning_cutoff : int, optional
        If pruning, the number of fittest offspring to keep. Must be supplied if `prune` is `True`. Defaults to `None`. 

        verbose : bool, optional
        Whether to show progress of the evolution as a progress bar. Defaults to True.

        Returns:
        -----------
        dict 
        The results of the evolution recorded in a dictionary with the following items:
        [key : type
        description]
        - 'fittest_individual' : np.ndarray 
        The fittest solution in decimal (phenotype) representation found by evolution 

        - 'best_overall_fitness' : float 
        The fitness value of the fittest solution found by evolution 

        - 'best_fitness_per_generation' : np.ndarray
        A record of the best fitness value per generation (or in other words a learning curve), of shape (N_gen,)

        - 'mean_fitness_per_generation' : np.ndarray 
        A record of the mean fitness value per generation (of the pruned population if applied), of shape (N_gen,)

        - 'median_fitness_per_generation' : np.ndarray
        A record of the median fitness value per generation (of the pruned population if applied), of shape (N_gen,)

        """
        pop_dec = pop_bin2dec(
            pop_bin = pop,
            N_bits_segment = self.N_bits_segment,
            decimal_acc = self.decimal_acc,
            offset = self.offset,
        )
        fitness_arr = self.fitness_func(pop_dec)
        best_fitness_per_gen   = [np.max(fitness_arr)]
        mean_fitness_per_gen   = [np.mean(fitness_arr)]
        median_fitness_per_gen = [np.median(fitness_arr)]

        if verbose:
            progress_bar = tqdm(total = N_gen, desc=f"[genepy] Evolution in progress", unit = "generations", file=sys.stdout,)

        for gen in range(N_gen):
            if switch is not None and gen >= switch:
                rank_selection = True
            elif switch is not None and gen < switch:
                rank_selection = False
            elif switch is None:
                rank_selection = False
            mating_pool = self.create_mating_pool(
                pop = pop,
                fitness_arr = fitness_arr,
                N_pairs = N_pairs,
                selection_type = selection_type,
                rank_selection = rank_selection,
            )
            offspring, fitness_arr = self.breed(
                mating_pool = mating_pool,
                crossover_type = crossover_type,
                prob_mut = prob_mut,
                prune = prune,
                pruning_cutoff = pruning_cutoff,
                return_fitness = True,
            )
            best_fitness_per_gen.append(np.max(fitness_arr))
            mean_fitness_per_gen.append(np.mean(fitness_arr))
            median_fitness_per_gen.append(np.median(fitness_arr))
            pop = offspring
            if verbose:
                progress_bar.update(1)

        indsort = np.argsort(fitness_arr)
        fittest_indiv = fitness_arr[indsort][-1]
        fittest_indiv_dec = bin2dec(fittest_indiv, self.N_bits_segment, self.decimal_acc)
        return {
            'fittest_individual': fittest_indiv_dec,
            'best_overall_fitness': fitness_arr[indsort][-1],
            'best_fitness_per_generation': np.array(best_fitness_per_gen),
            'mean_fitness_per_generation': np.array(mean_fitness_per_gen),
            'median_fitness_per_generation': np.array(median_fitness_per_gen),
        }