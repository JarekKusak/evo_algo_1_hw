import random
import os
import matplotlib.pyplot as plt

def select_parents(population, fitness_fn):
    #finess_of_individuals = [fitness_fn(i) for i in population] # Fitness proportionate selection

    fitness_of_individuals = []
    for i in range(len(population)):
        fitness_of_individuals.append(fitness_fn(population[i]))
    
    total_fitness = sum(fitness_of_individuals) # total fitness
    # guard against zero total fitness to avoid division by zero
    if total_fitness == 0:
        selection_probs = [1.0/len(population)] * len(population)
    else:
        selection_probs = [f / total_fitness for f in fitness_of_individuals] # selection probabilities

    # select two parents
    parent1 = random.choices(population, weights=selection_probs, k=1)[0]
    parent2 = random.choices(population, weights=selection_probs, k=1)[0] 
    return parent1, parent2

def crossover(parent1, parent2):
    # one-point crossover at a random point (not at the ends)
    if len(parent1) < 2:
        return parent1[:]
    point = random.randrange(1, len(parent1))
    child = parent1[:point] + parent2[point:]
    return child

''' rewritten functions from practicals (not used)
def cross(p1, p2):
    point = random.randrange(1, len(p1) - 1)
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2

def mutate(ind):
    return [1-i if random.random() < 0.01 else i for i in ind]

def mutation(pop):
    return [mutate(ind) if random.random() < 0.1 else ind[:] for ind in pop] # 10% chance to mutate each individual, else copy as is

def crossover(population):
    off = []
    if random.random() < 0.7: # Crossover chance
        return population[:] # No crossover, return population as is
    for (p1, p2) in zip(population[0::2], population[1::2]):
        o1, o2 = cross(p1, p2)
        off.extend([o1, o2])
    return off
'''

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate: # mutation chance
            individual[i] = 1 - individual[i]  # flip bit (assuming binary representation)
    return individual

def genetic_algorithm(population, fitness_fn, generations, mutation_rate=0.01, crossover_rate=0.9, elitism=2):
    """Run GA and return (best_individual, best_fitness_history).
    best_fitness_history records the best fitness at the start of each generation and after the last generation.
    """
    history = []
    for _ in range(generations):
        # log current best
        current_best = max(population, key=fitness_fn)
        history.append(fitness_fn(current_best))

        # sort and keep elites
        population = sorted(population, key=fitness_fn, reverse=True)
        next_generation = [ind[:] for ind in population[:elitism]]

        # produce the rest
        while len(next_generation) < len(population):
            parent1, parent2 = select_parents(population, fitness_fn)
            # crossover with probability
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1[:]
            # mutation
            mutate(child, mutation_rate)
            next_generation.append(child)

        population = next_generation

    # final log and return best
    best = max(population, key=fitness_fn)
    history.append(fitness_fn(best))
    return best, history

# fintness functions

# OneMax: count of 1s
def fitness_onemax(individual):
    return sum(individual)

# alternating pattern: score matches to 1010... and 0101..., take the max
def fitness_alternating(individual):
    n = len(individual)
    # build both target patterns on the fly
    score1 = 0  # matches to 1010...
    score2 = 0  # matches to 0101...
    for i in range(n):
        bit = individual[i]
        if bit == ((i+1) % 2): # 1010...
            score1 += 1
        if bit == (i % 2): # 0101...
            score2 += 1
    return score1 if score1 > score2 else score2

def init_population(pop_size, gene_len):
    return [[random.randint(0, 1) for _ in range(gene_len)] for _ in range(pop_size)]

def run_averaged(fitness_fn, gene_len, pop_size, generations, mutation_rate, crossover_rate, elitism, repeats):
    # accumulate histories and average them generation-wise
    totals = None
    for r in range(repeats):
        pop = init_population(pop_size, gene_len)
        _, hist = genetic_algorithm(pop, fitness_fn, generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate, elitism=elitism)
        if totals is None:
            totals = [0.0 for _ in hist]
        for i, v in enumerate(hist):
            totals[i] += v
    return [v / repeats for v in totals]

if __name__ == "__main__":
    random.seed(42)

    # baseline single run (demonstration)
    POP_SIZE = 50
    GENE_LEN = 40
    GENERATIONS = 100
    ELITISM = 2

    population = init_population(POP_SIZE, GENE_LEN)
    best_ind, hist = genetic_algorithm(population, fitness_onemax, generations=GENERATIONS, mutation_rate=0.01, crossover_rate=0.9, elitism=ELITISM)
    print("Demo OneMax – best individual:", best_ind)
    print("Demo OneMax – best fitness:", fitness_onemax(best_ind))

    # experiments for the assignment
    os.makedirs(".", exist_ok=True)

    # oneMax: sweep mutation rates
    mut_vals = [0.001, 0.01, 0.05]
    generations_axis = list(range(GENERATIONS + 1))

    curves_onemax = {}
    for m in mut_vals:
        curves_onemax[m] = run_averaged(
            fitness_fn=fitness_onemax,
            gene_len=GENE_LEN,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            mutation_rate=m,
            crossover_rate=0.9,
            elitism=ELITISM,
            repeats=10 # number of runs to average
        )

    # plot OneMax
    plt.figure()
    for m in mut_vals:
        plt.plot(generations_axis, curves_onemax[m], label=f"mutation_rate={m}")
    plt.xlabel("Generace")
    plt.ylabel("Nejlepší fitness (průměr přes běhy)")
    plt.title("GA konvergence – OneMax (různé míry mutace)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ga_onemax_convergence.png", dpi=150)

    # alternating pattern: sweep crossover rates
    cross_vals = [0.6, 0.8, 0.95]
    curves_alt = {}
    for c in cross_vals:
        curves_alt[c] = run_averaged(
            fitness_fn=fitness_alternating,
            gene_len=GENE_LEN,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            mutation_rate=0.01,
            crossover_rate=c,
            elitism=ELITISM,
            repeats=10
        )

    # plot Alternating
    plt.figure()
    for c in cross_vals:
        plt.plot(generations_axis, curves_alt[c], label=f"crossover_rate={c}")
    plt.xlabel("Generace")
    plt.ylabel("Nejlepší fitness (průměr přes běhy)")
    plt.title("GA konvergence – Střídavý vzor (různé míry křížení)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ga_alternating_convergence.png", dpi=150)

    print("\nSoubory vytvořeny v aktuálním adresáři:")
    print("- ga_onemax_convergence.png")
    print("- ga_alternating_convergence.png")