import random
import numpy as np
import functools

import utils
import os
import matplotlib.pyplot as plt

K = 10 #number of bins

POP_SIZE = 300  # population size (increased for better diversity)
MAX_GEN = 1200  # maximum number of generations (more time to converge)
# 300 * 1200 -> 360 000 evaluations of fitness
CX_PROB = 0.9   # crossover probability (slightly higher mixing)
MUT_PROB = 0.1  # mutation probability
MUT_FLIP_PROB = 0.06  # per-gene flip probability (a bit lower for late stability)

MUT_MOVE_PROB = 0.15  # probability of applying the guided move-one mutation to an individual

ELITISM = 3  # number of elites preserved each generation

REPEATS = 10 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'partition' # output directory for logs
EXP_ID = 'default' # the ID of this experiment (used to create log names)

# reads the input set of values of objects
def read_weights(filename):
    with open(filename) as f:
        return list(map(int, f.readlines()))

# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights, bins):
    bw = [0]*K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw

# Variant A fitness: adds sensitivity via std of bin weights
def fitness_ab(ind, weights, alpha=1.0, beta=0.3):
    bw = bin_weights(weights, ind)
    rng = max(bw) - min(bw)
    std = float(np.std(bw))
    fit_val = 1.0 / (1.0 + alpha * rng + beta * std)
    return utils.FitObjPair(fitness=fit_val, objective=rng)

# Optional advanced fitness (max-dev + L2). Used below in main as default for this file.
def fitness_adv(ind, weights, gamma=2.0):
    bw = bin_weights(weights, ind)
    T = sum(weights)/K
    # Primary target: minimize maximum deviation from target T
    max_dev = max(abs(w - T) for w in bw)
    # Secondary smoothing: sum of squared deviations
    l2 = sum((w - T)**2 for w in bw)
    # Monotonic transformation
    fit_val = 1.0 / (1.0 + max_dev + gamma * (l2 / (K * T * T)))
    # Keep objective compatible with logs: report range-like quantity (here max_dev)
    return utils.FitObjPair(fitness=fit_val, objective=max_dev)

# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

# implements uniform crossover of two individuals
# - each gene is independently taken from p1 or p2 with 50% probability
def uniform_cross(p1, p2):
    mask = [random.random() < 0.5 for _ in range(len(p1))]
    o1 = [p1[i] if mask[i] else p2[i] for i in range(len(p1))]
    o2 = [p2[i] if mask[i] else p1[i] for i in range(len(p1))]
    return o1, o2

# implements the "bit-flip" mutation of one individual
def flip_mutate(p, prob, upper):
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]

# guided mutation: move a single item from the heaviest bin to the lightest bin
# to reduce the maximum deviation of bin weights. If no beneficial move exists,
# return the individual unchanged.
def move_one_mutate(individual, item_weights):
    # create a working copy of the individual (assignment of items to bins)
    mutated_individual = individual[:]

    # compute total weights of each bin
    bin_total_weights = bin_weights(item_weights, mutated_individual)

    # find indices of the heaviest and lightest bins
    heaviest_bin_index = int(np.argmax(bin_total_weights))
    lightest_bin_index = int(np.argmin(bin_total_weights))

    # if all bins have the same weight, nothing to do
    if heaviest_bin_index == lightest_bin_index:
        return mutated_individual  # already perfectly balanced

    # collect indices of items that currently belong to the heaviest bin
    candidate_item_indices = []
    for item_index, bin_index in enumerate(mutated_individual):
        if bin_index == heaviest_bin_index:
            candidate_item_indices.append(item_index)

    if len(candidate_item_indices) == 0:
        return mutated_individual  # no items found in the heaviest bin

    # compute the target average bin weight
    target_bin_weight = sum(item_weights) / K

    # helper function to compute maximum deviation of bins from target weight
    def compute_max_deviation(current_bin_weights):
        max_dev = 0.0
        for bin_weight in current_bin_weights:
            deviation = abs(bin_weight - target_bin_weight)
            if deviation > max_dev:
                max_dev = deviation
        return max_dev

    # store current best objective value
    best_objective_value = compute_max_deviation(bin_total_weights)
    best_item_to_move_index = -1

    # try moving each candidate item from the heaviest bin to the lightest bin
    # evaluate how much it improves the maximum deviation
    for item_index in candidate_item_indices:
        item_weight = item_weights[item_index]

        # temporarily modify bin weights
        old_heavy_weight = bin_total_weights[heaviest_bin_index]
        old_light_weight = bin_total_weights[lightest_bin_index]

        bin_total_weights[heaviest_bin_index] = old_heavy_weight - item_weight
        bin_total_weights[lightest_bin_index] = old_light_weight + item_weight

        # evaluate new objective value
        current_objective_value = compute_max_deviation(bin_total_weights)

        # restore bin weights after evaluation
        bin_total_weights[heaviest_bin_index] = old_heavy_weight
        bin_total_weights[lightest_bin_index] = old_light_weight

        # keep track of the best move (lowest objective)
        if current_objective_value < best_objective_value:
            best_objective_value = current_objective_value
            best_item_to_move_index = item_index

    # if an improving move was found, apply it
    if best_item_to_move_index != -1:
        mutated_individual[best_item_to_move_index] = lightest_bin_index

    # return possibly modified individual
    return mutated_individual

# tournament selection
def tournament_selection(pop, fits, k, tour_size=3, p=0.8):
    selected = []
    n = len(pop)
    for _ in range(k):
        candidate_index = [random.randrange(0, n) for _ in range(tour_size)]
        candidates = sorted(candidate_index, key=lambda i: fits[i], reverse=True)
        if random.random() < p:
            chosen = candidates[0]
        else:
            chosen = random.choice(candidates[1:]) if len(candidates) > 1 else candidates[0]
        selected.append(pop[chosen])
    return selected

# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    return random.choices(pop, fits, k=k)

# applies a list of genetic operators (functions with 1 argument - population) 
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None, elitism=1):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]

        # keep elites
        elites_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:elitism]
        elites = [pop[i][:] for i in elites_idx]

        # selection + variation
        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)

        # replace part of offspring with elites
        pop = offspring[:POP_SIZE-elitism] + elites
    return pop

if __name__ == '__main__':
    # === Dataset (easy) ===
    weights = read_weights('inputs/partition-easy.txt')

    # Output directory for this custom version
    MY_DIR = 'partition_my_version'
    os.makedirs(MY_DIR, exist_ok=True)

    # selection (tournament), operators (uniform CX + flip + move-one), fitness (fitness_ab)
    cr_ind = functools.partial(create_ind, ind_len=len(weights))
    fit = functools.partial(fitness_ab, weights=weights)
    mate_sel = functools.partial(tournament_selection, tour_size=3, p=0.8)

    xover = functools.partial(crossover, cross=uniform_cross, cx_prob=CX_PROB)
    mut_flip = functools.partial(mutation, mut_prob=MUT_PROB,
                                 mutate=functools.partial(flip_mutate, prob=MUT_FLIP_PROB, upper=K))
    mut_move = functools.partial(mutation, mut_prob=MUT_MOVE_PROB,
                                 mutate=functools.partial(move_one_mutate, item_weights=weights))

    import multiprocessing
    pool = multiprocessing.Pool()

    EXP_ID = 'my_version'
    best_inds = []

    for run in range(REPEATS):
        # log to MY_DIR with EXP_ID
        log = utils.Log(MY_DIR, EXP_ID, run, write_immediately=True, print_frequency=5)
        pop = create_pop(POP_SIZE, cr_ind)

        pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut_flip, mut_move], mate_sel,
                                     map_fn=pool.map, log=log, elitism=ELITISM)

        bi = max(pop, key=fit)
        best_inds.append(bi)

        # save per-run .best
        with open(f'{MY_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            for w, b in zip(weights, bi):
                f.write(f'{w} {b}\n')

    # Summarize & plot for this experiment
    utils.summarize_experiment(MY_DIR, EXP_ID)
    evals, lower, mean, upper = utils.get_plot_data(MY_DIR, EXP_ID)
    plt.figure(figsize=(12, 8))
    utils.plot_experiment(evals, lower, mean, upper, legend_name='My version (uniform + move-one + tournament)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MY_DIR, f'{EXP_ID}.png'), dpi=150)
    plt.close()

    # Save consolidated best across all runs
    objs = []
    for bi in best_inds:
        fo = fit(bi).objective
        objs.append(fo)
    best_idx = int(np.argmin(objs))
    best_overall = best_inds[best_idx]
    with open(f'{MY_DIR}/{EXP_ID}.best', 'w') as f:
        for w, b in zip(weights, best_overall):
            f.write(f'{w} {b}\n')

    print(f'[my_partition] Best difference across runs = {min(objs)} '
          f'(saved to {MY_DIR}/{EXP_ID}.best)')
    print(f'Plot saved to {MY_DIR}/{EXP_ID}.png')
