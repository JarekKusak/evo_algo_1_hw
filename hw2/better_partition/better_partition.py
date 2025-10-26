import random
import numpy as np
import functools
from typing import List

import utils

K = 10 #number of bins

POP_SIZE = 100 # population size
MAX_GEN = 500 # maximum number of generations
CX_PROB = 0.8 # crossover probability
MUT_PROB = 0.2 # mutation probability
MUT_FLIP_PROB = 0.1 # probability of chaninging value during mutation

REPEATS = 10 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'better_partition' # output directory for logs
EXP_ID = 'default' # the ID of this experiment (used to create log names)

# mixed initialization (our minimal change)
USE_MIXED_INIT = True
MIXED_INIT_CFG = (0.35, 0.25, 0.02)  # greedy_frac, noisy_greedy_frac, noise_rate
# extra diversified seeds (fractions)
JITTERED_FRAC = 0.25
SOFT_FRAC = 0.10
REPAIRED_FRAC = 0.10
# their knobs
JITTER_EPS = 0.08
SOFT_TEMP = 6.0

REPAIR_STEPS = 250

# minimal survivor elitism (to help escape plateaus)
ELITISM = 1  # keep top-1 parent each generation; set 0 to disable

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

def greedy_seed(weights: List[int]) -> List[int]:
    """Best-Fit Decreasing: assign items (sorted desc) to currently lightest bin."""
    idx_sorted = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    bins_sum = [0]*K
    assign = [0]*len(weights)
    for i in idx_sorted:
        currently_lightest_bin = min(range(K), key=lambda b: bins_sum[b])
        assign[i] = currently_lightest_bin
        bins_sum[currently_lightest_bin] += weights[i]
    return assign

def greedy_seed_jitter(weights: List[int], eps: float = 0.05) -> List[int]:
    """Greedy with jittered sort key: sort by w * (1 + noise in [-eps, eps])."""
    jitter = [1.0 + random.uniform(-eps, eps) for _ in weights]
    idx_sorted = sorted(range(len(weights)), key=lambda i: weights[i] * jitter[i], reverse=True)
    bins_sum = [0]*K
    assign = [0]*len(weights)
    for i in idx_sorted:
        currently_lightest_bin = min(range(K), key=lambda b: bins_sum[b])
        assign[i] = currently_lightest_bin
        bins_sum[currently_lightest_bin] += weights[i]
    return assign

def greedy_soft_seed(weights: List[int], temp: float = 5.0) -> List[int]:
    """Soft-min greedy: assign to bins with probability ~ exp(-load/temp). Lower temp -> greedier."""
    if temp <= 0:
        temp = 1e-6
    idx_sorted = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    bins_sum = [0.0]*K
    assign = [0]*len(weights)
    for i in idx_sorted:
        # compute softmin probabilities
        exps = [np.exp(-load / temp) for load in bins_sum]
        s = sum(exps)
        r = random.random() * s
        acc = 0.0
        chosen = 0
        for b, e in enumerate(exps):
            acc += e
            if acc >= r:
                chosen = b
                break
        assign[i] = chosen
        bins_sum[chosen] += float(weights[i])
    return assign

def greedy_noise(ind: List[int], rate: float, upper: int = K) -> List[int]:
    """Make a small random scratch in a greedy individual to create diversity."""
    out = ind[:]
    for i in range(len(out)):
        if random.random() < rate:
            out[i] = random.randrange(0, upper)
    return out

def local_balance_steps(ind: List[int], weights: List[int], steps: int = 200) -> List[int]:
    """Do a few first-improvement moves from heaviest bin to lightest if they reduce range.
    Used only at initialization to diversify basins without changing genetic operators.
    """
    assign = ind[:]
    for _ in range(steps):
        bw = bin_weights(weights, assign)
        h = int(np.argmax(bw))
        l = int(np.argmin(bw))
        best_delta = 0
        best_idx = -1
        # try moving a single item h->l
        for i, (w, b) in enumerate(zip(weights, assign)): # iterate items
            if b != h:
                continue
            new_h = bw[h] - w
            new_l = bw[l] + w
            # compute new range if we did the move
            new_range = max(max(new_h, *(bw[j] for j in range(K) if j not in (h, l))), new_l) - \
                        min(min(new_h, *(bw[j] for j in range(K) if j not in (h, l))), new_l)
            cur_range = max(bw) - min(bw)
            delta = cur_range - new_range
            if delta > best_delta:
                best_delta = delta
                best_idx = i
        if best_delta > 0 and best_idx >= 0:
            assign[best_idx] = l
        else:
            break
    return assign

def create_pop_mixed(pop_size: int, rnd_create, weights: List[int],
                     greedy_frac: float = 0.3,
                     noisy_greedy_frac: float = 0.2,
                     noise_rate: float = 0.02,
                     jittered_greedy_frac: float = 0.0,
                     soft_greedy_frac: float = 0.0,
                     repaired_frac: float = 0.0,
                     jitter_eps: float = 0.05,
                     soft_temp: float = 5.0,
                     repair_steps: int = 200) -> List[List[int]]:
    """Mix of random, greedy and diversified greedy individuals.

    Additional fractions (default 0.0 so it's backward compatible):
      - jittered_greedy_frac: greedy with jittered sort to vary early placements
      - soft_greedy_frac: soft-min greedy controlled by temperature
      - repaired_frac: apply a few local-balance steps to a greedy-like seed
    """
    '''
    Jak inicializujeme populaci:
    - greedy_frac: Best-Fit Decreasing (BFD) jedinci (BFD přiřazuje objekty seřazené sestupně do aktuálně nejlehčího binu)
    - noisy_greedy_frac: BFD jedinci s malým náhodným šumem (každý objekt má s pravděpodobností noise_rate přiřazen náhodný bin)
    - jittered_greedy_frac: BFD jedinci, kde je pořadí objektů určeno s přidaným náhodným šumem (každý objekt má náhodný faktor v rozsahu [-jitter_eps, jitter_eps], který ovlivňuje jeho váhu při řazení)
    - soft_greedy_frac: jedinci generovaní pomocí "soft-min" greedy přístupu, kde je výběr binu řízen pravděpodobnostmi závislými na aktuálním zatížení binů a teplotou soft_temp (nižší teplota znamená více "greedier" výběr)
    - repaired_frac: jedinci, kteří jsou nejprve generováni pomocí jittered greedy metody a poté jsou vylepšeni pomocí několika kroků lokálního vyvážení (local balance), což pomáhá snížit rozdíl mezi nejvíce a nejméně zatíženými biny
    - zbytek populace je vyplněn náhodně generovanými jedinci
    '''

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # clamp fractions to [0, 1]
    greedy_frac = clamp01(greedy_frac)
    noisy_greedy_frac = clamp01(noisy_greedy_frac)
    jittered_greedy_frac = clamp01(jittered_greedy_frac)
    soft_greedy_frac = clamp01(soft_greedy_frac)
    repaired_frac = clamp01(repaired_frac)

    # determine counts of each type (dependent on fractions)
    n_greedy = int(pop_size * greedy_frac) 
    n_noisy  = int(pop_size * noisy_greedy_frac)
    n_jit    = int(pop_size * jittered_greedy_frac)
    n_soft   = int(pop_size * soft_greedy_frac)
    n_rep    = int(pop_size * repaired_frac)
    n_used   = n_greedy + n_noisy + n_jit + n_soft + n_rep
    n_rand   = max(0, pop_size - n_used)

    pop: List[List[int]] = []
    base_greedy = greedy_seed(weights)

    # create individuals of each type
    # (BFD = Best Fit Decreasing)
    for _ in range(n_greedy):
        pop.append(base_greedy[:]) # BFD základní greedy: hladově přidává položky do nejlehčího binu
    for _ in range(n_noisy):
        pop.append(greedy_noise(base_greedy, noise_rate)) # greedy s náhodným šumem: s malou pravděpodobností přiřadí položky do náhodných binů (upravuje konečné přiřazení)
    for _ in range(n_jit):
        pop.append(greedy_seed_jitter(weights, jitter_eps)) # jittered greedy: BFD jedinci, kde je pořadí objektů určeno s přidaným náhodným šumem
    for _ in range(n_soft):
        pop.append(greedy_soft_seed(weights, soft_temp)) # soft greedy: jedinci generovaní pomocí "soft-min" greedy přístupu
    for _ in range(n_rep):
        seed = greedy_seed_jitter(weights, jitter_eps) # repaired: jedinci, kteří jsou nejprve generováni pomocí jittered greedy metody
        pop.append(local_balance_steps(seed, weights, steps=repair_steps))
    for _ in range(n_rand):
        pop.append(rnd_create())

    # shuffle the population to mix different types
    random.shuffle(pop)
    return pop

# the fitness function
def fitness(ind, weights):
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1/(max(bw) - min(bw) + 1), 
                            objective=max(bw) - min(bw))

# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    return random.choices(pop, fits, k=k)

# the tournament selection (configurable via functools.partial)
def tournament_selection(pop, fits, k, tour_size=2, p=0.7):
    selected = []
    n = len(pop)
    for _ in range(k):
        cand_idx = [random.randrange(0, n) for _ in range(tour_size)]
        cand_idx.sort(key=lambda i: fits[i], reverse=True)
        if random.random() < p:
            chosen = cand_idx[0]
        else:
            chosen = random.choice(cand_idx[1:]) if len(cand_idx) > 1 else cand_idx[0]
        selected.append(pop[chosen])
    return selected

# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

# implements the "bit-flip" mutation of one individual
def flip_mutate(p, prob, upper):
    return [random.randrange(0, upper) if random.random() < prob else i for i in p]

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
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        # minimal elitism: keep best ELITISM parents
        if ELITISM > 0:
            parent_pairs = list(zip(pop, fits))
            parent_pairs.sort(key=lambda x: x[1], reverse=True)
            elites = [p for p, _ in parent_pairs[:ELITISM]]
            offspring[-ELITISM:] = [e[:] for e in elites]
        pop = offspring[:]

    return pop

if __name__ == '__main__':
    # dataset (easy)
    weights = read_weights('inputs/partition-easy.txt')

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=len(weights))
    fit = functools.partial(fitness, weights=weights)
    xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
    mut = functools.partial(mutation, mut_prob=MUT_PROB, 
                            mutate=functools.partial(flip_mutate, prob=MUT_FLIP_PROB, upper=K))

    # selection: milder pressure than roulette to preserve diversity
    mate_sel = functools.partial(tournament_selection, tour_size=2, p=0.7)

    # we can use multiprocessing to evaluate fitness in parallel
    import multiprocessing
    pool = multiprocessing.Pool()

    import matplotlib.pyplot as plt

    def run_one_experiment(exp_id: str, use_mixed_init: bool):
        best_inds = []
        for run in range(REPEATS):
            log = utils.Log(OUT_DIR, exp_id, run,
                            write_immediately=True, print_frequency=5)
            # population init: smart (mixed) vs random
            if use_mixed_init:
                gfrac, ngfrac, noise = MIXED_INIT_CFG
                pop = create_pop_mixed(
                    POP_SIZE, cr_ind, weights,
                    greedy_frac=gfrac,
                    noisy_greedy_frac=ngfrac,
                    noise_rate=noise,
                    jittered_greedy_frac=JITTERED_FRAC,
                    soft_greedy_frac=SOFT_FRAC,
                    repaired_frac=REPAIRED_FRAC,
                    jitter_eps=JITTER_EPS,
                    soft_temp=SOFT_TEMP,
                    repair_steps=REPAIR_STEPS,
                )
            else:
                pop = create_pop(POP_SIZE, cr_ind)

            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], mate_sel, map_fn=pool.map, log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)
            with open(f'{OUT_DIR}/{exp_id}_{run}.best', 'w') as f:
                for w, b in zip(weights, bi):
                    f.write(f'{w} {b}\n')
        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'[{exp_id}] Run {i}: difference = {fit(bi).objective}, bin weights = {bin_weights(weights, bi)}')

        # highlight the best run across repeats
        objs = [fit(bi).objective for bi in best_inds]
        best_idx = int(np.argmin(objs))
        print(f'[{exp_id}] BEST run = {best_idx}: difference = {objs[best_idx]}, bin weights = {bin_weights(weights, best_inds[best_idx])}')

        # summarize logs for this experiment
        utils.summarize_experiment(OUT_DIR, exp_id)

    # --- Run two experiments: smart init vs random init ---
    run_one_experiment('smart', use_mixed_init=True)
    run_one_experiment('random', use_mixed_init=False)

    # Plot both experiments together (quartiles) + overlay BEST run (lower envelope)
    plt.figure(figsize=(12, 8))
    utils.plot_experiments(OUT_DIR, ['smart', 'random'],
                            rename_dict={'smart': 'Smart init', 'random': 'Random init'},
                            stat_type='objective')

    # overlay the best (minimum) curve per generation for each experiment
    for exp_id, style, label in [
        ('smart', '-', 'Smart init – BEST'),
        ('random', '--', 'Random init – BEST'),
    ]:
        evals, lower, q25, mean, q75, upper = utils.get_experiment_stats(OUT_DIR, exp_id, stat_type='objective')
        plt.plot(evals, lower, linestyle=style, linewidth=2.0, label=label)

    plt.legend()
    plt.tight_layout()
    plt.show()
