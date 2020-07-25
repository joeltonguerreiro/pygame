import operator
import random

import numpy as np
import pandas as pd


def initial_population(population_size, input_size, hidden_size, output_size):
    population = []

    for i in range(0, population_size):
        individual = [
            np.random.uniform(-1., 1., (input_size, hidden_size)),
            np.random.uniform(-1., 1., (hidden_size, hidden_size)),
            np.random.uniform(-1., 1., (hidden_size, output_size))
        ]

        population.append(individual)

    return population


def rank_individuals(population, fitness):

    fitness_results = fitness(population).get_fitness()

    rank = sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=False)

    print('rank', rank)

    return rank


def selection(population_ranked, elite_size):

    selection_results = []

    for i in range(0, elite_size):
        selection_results.append(population_ranked[i][0])
        del population_ranked[i]

    df = pd.DataFrame(np.array(population_ranked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    top_percent = int(len(population_ranked) * 0.3)

    firsts = population_ranked[:top_percent]

    length = len(firsts)

    while len(selection_results) < length:
        pick = 100 * random.random()

        for j in range(0, length):
            if pick <= df.iat[j, 3]:
                if population_ranked[j][0] not in selection_results:
                    selection_results.append(population_ranked[j][0])
                    break

    return selection_results


def breed(parent1, parent2):

    child = []

    # a reproducao vai ser diferente
    # obter cada array de pesos
    # mesclar as propriedades de cada array
    # adicionar ao filho cada array

    # weights = []

    for i in range(0, len(parent1)):

        sep = random.randint(1, len(parent1[i]) - 1)

        weights = []

        child_p1 = parent1[i][:sep]
        child_p2 = parent2[i][sep:]

        for j in child_p1:
            weights.append(j)

        for k in child_p2:
            weights.append(k)

        child.append(weights)

        '''parentI = random.randint(1, 2)

        if parentI == 1:
            weights.append(parent1[i])
        if (parentI == 2):
            weights.append(parent2[i])

        child = weights'''

    return child


def breed_population(matingpool, elite_size, current_generation):
    children = []

    # length = len(matingpool) - eliteSize
    length = len(current_generation) - elite_size

    for i in range(0, elite_size):
        children.append(matingpool[i])

    pool = random.sample(matingpool, len(matingpool))

    # sep = random.randint(1, len(matingpool))
    # pool = matingpool[:sep]

    # pool = random.sample(matingpool, length)

    for i in range(0, length):
        index1 = random.randint(0, len(pool) - 1)
        index2 = random.randint(0, len(pool) - 1)
        parent1 = pool[index1]
        parent2 = pool[index2]

        child = breed(parent1, parent2)

        children.append(child)

    return children


def mutate(individual, mutation_rate):
    for i in np.arange(len(individual)):
        for neuron in range(len(individual[i])):
            if random.random() < mutation_rate:
                new_weight = []

                for j in range(len(individual[i][neuron])):
                    new_weight.append(random.uniform(-1., 1))

                # print('mutacao', len(new_weight))
                individual[i][neuron] = new_weight

    return individual


'''def mutate(individual, mutation_rate):
    i = 1
    for neuron in range(len(individual[i])):
        if random.random() < mutation_rate:
            new_weight = []

            n1 = random.randint(0, len(individual[i][neuron]) - 1)
            n2 = random.randint(0, len(individual[i][neuron]) - 1)

            for j in range(len(individual[i][neuron])):
                if j == n1:
                    new_weight.append(individual[i][neuron][n2])
                elif j == n2:
                    new_weight.append(individual[i][neuron][n1])
                else:
                    new_weight.append(random.uniform(-1., 1))

            individual[i][neuron] = new_weight

    return individual'''


def mutate_population(population, mutation_rate):
    mutated_population = []

    for individual in range(0, len(population)):
        mutated_individual = mutate(population[individual], mutation_rate)
        mutated_population.append(mutated_individual)

    return mutated_population


def mating_pool(population, selection_results):
    matingpool = []

    for i in range(0, len(selection_results)):
        index = selection_results[i]
        matingpool.append(population[index])

    return matingpool


def next_generation(current_generation, elite_size, mutation_rate, fitness):
    population_ranked = rank_individuals(current_generation, fitness)

    selection_results = selection(population_ranked, elite_size)

    matingpool = mating_pool(current_generation, selection_results)

    children = breed_population(matingpool, elite_size, current_generation)

    return mutate_population(children, mutation_rate)


def save_weights(filename, population):
    weights = np.array([])

    for ind in population:
        for bias in ind:
            weights = np.array(np.append(weights, bias))

    np.savetxt(filename, weights)


def parse_weights_population(new_data, input_size, hidden_size, output_size):
    n1_end = input_size * hidden_size
    n2_start = input_size * hidden_size
    n2_end = (input_size + hidden_size) * hidden_size
    n3_start = (input_size + hidden_size) * hidden_size

    population = []

    for ind in new_data:
        n1 = ind[:n1_end]
        n2 = ind[n2_start:n2_end]
        n3 = ind[n3_start:]

        new_array = [np.split(n1, input_size),
                     np.split(n2, hidden_size),
                     np.split(n3, hidden_size)]

        population.append(new_array)

    return population


class GeneticAlgorithm:
    def __init__(self, population_size, elite_size, mutation_rate, generations, fitness):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.current_population = []
        self.fitness = fitness

    def start(self):
        input_size = 4
        hidden_size = 6
        output_size = 4
        self.current_population = initial_population(self.population_size, input_size, hidden_size, output_size)

        load_weights = True

        if load_weights:
            data = np.loadtxt('data_weights.csv')

            new_data = np.split(data, self.population_size)

            self.current_population = parse_weights_population(new_data, input_size, hidden_size, output_size)

        else:
            self.current_population = initial_population(self.population_size, input_size, hidden_size, output_size)

        for i in range(0, self.generations):
            print('generation: ', i)
            # results.incrementGenerations()
            self.current_population = next_generation(self.current_population,
                                                      self.elite_size,
                                                      self.mutation_rate,
                                                      self.fitness
                                                      )

            is_to_save_weights = True

            if is_to_save_weights:
                save_weights('data_weights.csv', self.current_population)

        # best_individual_index = self.rankIndividuals(self.currentPopulation)[0][0]
        # best_individual = self.currentPopulation[best_individual_index]
        # print(best_individual)



