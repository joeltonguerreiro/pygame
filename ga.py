import numpy as np
import operator
import pandas as pd
import random


def initial_population(population_size, input_size, hidden_size, output_size):
    population = []

    for i in range(0, population_size):
        individual = [np.random.uniform(-1., 1., (input_size, hidden_size)),
                      np.random.uniform(-1., 1., (hidden_size, hidden_size)),
                      np.random.uniform(-1., 1., (hidden_size, output_size))]

        population.append(individual)

    return population


def rank_individuals(population, fitness):

    fitness_results = fitness(population).get_fitness()

    rank = sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=False)

    return rank


def selection(population_ranked, elite_size):

    selection_results = []
    df = pd.DataFrame(np.array(population_ranked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(population_ranked[i][0])

    top_percent = int(len(population_ranked) * 0.3)

    firsts = population_ranked[:top_percent]

    length = len(firsts)

    while len(selection_results) < length:
        pick = 100 * random.random()

        for j in range(0, length):
            if pick <= df.iat[j, 3]:
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
        # print('mutate weight', i)
        for neuron in range(len(individual[i])):
            if random.random() < mutation_rate:
                new_weight = []
                for j in range(len(individual[i][neuron])):
                    new_weight.append(random.uniform(-1., 1))
                    # print('neuronio', individual[i][neuron])

                print('mutacao', new_weight)

                individual[i][neuron] = new_weight

    return individual


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

        for i in range(0, self.generations):
            print('generation: ', i)
            # results.incrementGenerations()
            self.current_population = next_generation(self.current_population,
                                                      self.elite_size,
                                                      self.mutation_rate,
                                                      self.fitness
                                                      )

        # best_individual_index = self.rankIndividuals(self.currentPopulation)[0][0]
        # best_individual = self.currentPopulation[best_individual_index]
        # print(best_individual)

