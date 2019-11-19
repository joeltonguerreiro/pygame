import numpy as np
import operator
import pandas as pd
import random

import pygame

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
WIDTH, HEIGHT = 600, 300

SQUARE_WIDTH, SQUARE_HEIGHT = 5, 5
WALL_WIDTH, WALL_HEIGHT = 20, 20

velocity = 0.16
kill_line_velocity = 0.1

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

levels = [
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BWWWWWWWWW                   B",
    "BW  I    W                   B",
    "BW       W                   B",
    "BW       W                   B",
    "BW       W                   B",
    "BW       WWWWWWWWWWWWWWWWWWWWB",
    "BW                          WB",
    "BW                          WB",
    "BWWWWWWWWWWWWWWWWWWW        WB",
    "B                  W        WB",
    "B                  W        WB",
    "B                  W     E  WB",
    "B                  WWWWWWWWWWB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
]

left = 0
top = 0

walls = []
borders = []
exit = 0
pos_init = (0, 0)

for row in levels:
    for col in row:
        if col == 'W':
            wall = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
            walls.append(wall)
        if col == 'B':
            border = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
            borders.append(border)
        if col == 'E':
            exit = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
        left += WALL_WIDTH
        if col == 'I':
            pos_init = left, top
    top += WALL_HEIGHT
    if left >= WIDTH:
        left = 0

font = pygame.font.Font('freesansbold.ttf', 12)


class Results:
    def __init__(self):
        self.winners = 0
        self.generations = 0

    def incrementWinners(self, winner):
        self.winners += winner

    def getWinners(self):
        return str(self.winners)

    def resetWinners(self):
        self.winners = 0

    def incrementGenerations(self):
        self.generations += 1

    def getGenerations(self):
        return self.generations


results = Results()


class Fitness:
    def __init__(self, population):
        self.population = population
        self.bias = 1

        self.listSquares = []
        # 0 weights
        # 1 rect
        # 2 alive
        # 3 color
        # 4 distance

        self.fitnessResults = {}
        self.hasSquaresAlive = True
        self.bestDistance = 1000000

    def sigmoid(self, x, derivative=False):
        return x * (1.0 - x) if derivative else 1.0 / (1.0 + np.exp(-x))

    # calcular a ação a ser executada seguindo a maior probabilidade
    def feedforward(self, individual, input=None):

        input_array = np.array(input, ndmin=2)

        w1 = individual[0]
        w2 = individual[1]
        w3 = individual[2]

        layer1 = self.sigmoid(np.dot(input_array, w1) + self.bias)
        layer2 = self.sigmoid(np.dot(layer1, w2) + self.bias)
        output = self.sigmoid(np.dot(layer2, w3) + self.bias)

        return output;

    def calDistance(self):

        results.resetWinners()

        kill_line = pygame.Rect(0, 0, 3, 300)

        for i in range(0, len(self.population)):
            square = [self.population[i], pygame.Rect(pos_init[0], pos_init[1], SQUARE_WIDTH, SQUARE_HEIGHT), True,
                      (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 0]

            self.listSquares.append(square)

        while self.hasSquaresAlive:

            screen.fill(BLACK)

            dt = clock.tick(60)

            if kill_line.centerx == 600:
                kill_line.centerx = 0
                self.hasSquaresAlive = False
                continue

            kill_line.move_ip(dt * kill_line_velocity, 0)
            pygame.draw.rect(screen, RED, kill_line)

            for wall in walls:
                pygame.draw.rect(screen, WHITE, wall)

            for border in borders:
                pygame.draw.rect(screen, RED, border)

            pygame.draw.rect(screen, GREEN, exit)

            for index in range(0, len(self.listSquares)):

                if not self.listSquares[index][2]:
                    continue

                distance_top = distance_bottom = distance_right = distance_left = 600

                closerRight = closerBottom = closerLeft = closerTop = 0

                for wall in walls:

                    square = self.listSquares[index][1]

                    # o quadrado é menor que as paredes, então ele sempre estará
                    # contido em uma parede, os testes abaixo verificam esse caso

                    if square.left > wall.left and square.right < wall.right:
                        if square.top > wall.bottom:
                            if abs(square.top - wall.bottom) <= distance_top:
                                distance_top = abs(square.top - wall.bottom)
                                closerTop = wall

                        if square.bottom < wall.top:
                            if abs(square.bottom - wall.top) <= distance_bottom:
                                distance_bottom = abs(square.bottom - wall.top)
                                closerBottom = wall

                    if square.top > wall.top and square.bottom < wall.bottom:
                        if square.right < wall.left:
                            if abs(square.right - wall.left) <= distance_right:
                                distance_right = abs(square.right - wall.left)
                                closerRight = wall

                        if square.left > wall.right:
                            if abs(square.left - wall.right) <= distance_left:
                                distance_left = abs(square.left - wall.right)
                                closerLeft = wall

                x1, y1 = square.center

                if closerRight != 0:
                    r_x2, r_y2 = closerRight.center
                    pygame.draw.line(screen, RED, (x1, y1), (r_x2, r_y2), 1)

                if closerBottom != 0:
                    b_x2, b_y2 = closerBottom.center
                    pygame.draw.line(screen, RED, (x1, y1), (b_x2, b_y2), 1)

                if closerLeft != 0:
                    l_x2, l_y2 = closerLeft.center
                    pygame.draw.line(screen, RED, (x1, y1), (l_x2, l_y2), 1)

                if closerTop != 0:
                    t_x2, t_y2 = closerTop.center
                    pygame.draw.line(screen, RED, (x1, y1), (t_x2, t_y2), 1)

                input_array = distance_right, distance_bottom, distance_left, distance_top

                output = self.feedforward(self.listSquares[index][0], input_array)

                action = np.argmax(output)

                # 0 direita
                # 1 baixo
                # 2 esquerda
                # 3 cima

                pos_x = pos_y = 0

                if action == 0:
                    pos_x = velocity * dt

                if action == 1:
                    pos_y = (velocity * dt)

                if action == 2:
                    pos_x = -(velocity * dt)

                if action == 3:
                    pos_y = -(velocity * dt)

                self.listSquares[index][1].move_ip(pos_x, pos_y)
                pygame.draw.rect(screen, self.listSquares[index][3], self.listSquares[index][1])

                if self.listSquares[index][1].colliderect(kill_line):
                    self.listSquares[index][2] = False

                for wall in walls:
                    if self.listSquares[index][1].colliderect(wall):
                        # print('bateu')
                        self.listSquares[index][2] = False

                for border in borders:
                    if self.listSquares[index][1].colliderect(border):
                        # print('bateu')
                        self.listSquares[index][2] = False

                # init_x, init_y = WIDTH // 2, HEIGHT // 2
                init_x, init_y = exit.centerx, exit.centery
                current_x, current_y = self.listSquares[index][1].centerx, self.listSquares[index][1].centery

                xDis = abs(current_x - init_x)
                # calcula a distancia dos eixos y

                yDis = abs(current_y - init_y)
                # eleva as diferenças ao quadrado, soma e calcula a raiz

                distance = np.sqrt((xDis ** 2) + (yDis ** 2))

                if distance < self.bestDistance:
                    self.bestDistance = distance

                self.listSquares[index][4] = distance

                if self.listSquares[index][1].colliderect(exit):
                    results.incrementWinners(1)

            alives = 0

            for alive in self.listSquares:
                if alive[2]:
                    alives += 1

            if alives == 0:
                self.hasSquaresAlive = False

            generations_text = 'geracao: ' + str(results.getGenerations())
            generations_font = font.render(generations_text, True, WHITE)
            generations_rect = generations_font.get_rect()
            generations_rect.center = (50, HEIGHT - 80)

            screen.blit(generations_font, generations_rect)

            alives_text = 'vivos: ' + str(alives)
            alives_font = font.render(alives_text, True, WHITE)
            alives_rect = alives_font.get_rect()
            alives_rect.center = (50, HEIGHT - 60)

            screen.blit(alives_font, alives_rect)

            winners_text = 'vencedores: ' + str(results.getWinners())
            winners_font = font.render(winners_text, True, WHITE)
            winners_rect = winners_font.get_rect()
            winners_rect.center = (50, HEIGHT - 40)

            screen.blit(winners_font, winners_rect)

            pygame.display.flip()

        return True

    def getPathFitness(self):

        self.calDistance()

        print('bestDistance', self.bestDistance)

        for index in range(0, len(self.listSquares)):
            distance = self.listSquares[index][4]

            fitness = 1 / distance

            self.fitnessResults[index] = fitness

        return self.fitnessResults


class GeneticAlgorithm:
    def __init__(self, populationSize, eliteSize, mutationRate, generations):
        # self.population = population
        self.populationSize = populationSize
        self.eliteSize = eliteSize
        self.mutationRate = mutationRate
        self.generations = generations
        self.currentPopulation = []
        self.bestFitnessMovimentations = 1
        self.bestDistance = 0

    def start(self):
        self.currentPopulation = self.initialPopulation(self.populationSize, 4, 2, 4)

        for i in range(0, self.generations):
            print('generation: ', i)
            results.incrementGenerations()
            self.currentPopulation = self.nextGeneration(self.currentPopulation, self.eliteSize, self.mutationRate)

        bestIndividualIndex = self.rankIndividuals(self.currentPopulation)[0][0]
        bestIndividual = self.currentPopulation[bestIndividualIndex]
        print(bestIndividual)

    def createPath(self, listDirections):
        return random.sample(listDirections, len(listDirections))

    def initialPopulation(self, populationSize, input_size, hidden_size, output_size):
        population = []

        for i in range(0, populationSize):
            individual = [np.random.uniform(-1., 1., (input_size, hidden_size)),
                          np.random.uniform(-1., 1., (hidden_size, hidden_size)),
                          np.random.uniform(-1., 1., (hidden_size, output_size))]

            population.append(individual)

        return population

    def rankIndividuals(self, population):

        fitnessResults = Fitness(population).getPathFitness()

        rank = sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

        return rank

    def selection(self, populationRanked, eliteSize):

        selectionResults = []
        df = pd.DataFrame(np.array(populationRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

        for i in range(0, eliteSize):
            selectionResults.append(populationRanked[i][0])

        top30Percent = int(len(populationRanked) * 0.3)

        firsts = populationRanked[:top30Percent]

        length = len(firsts)

        while len(selectionResults) < length:
            pick = 100 * random.random()

            for j in range(0, length):
                if pick <= df.iat[j, 3]:
                    selectionResults.append(populationRanked[j][0])
                    break

        return selectionResults

    def matingPool(self, population, selectionResults):
        matingpool = []

        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])

        return matingpool

    def breed(self, parent1, parent2):

        child = []
        childP1 = []
        childP2 = []

        # a reproducao vai ser diferente
        # obter cada array de pesos
        # mesclar as propriedades de cada array
        # adicionar ao filho cada array

        # weights = []

        for i in range(0, len(parent1)):

            sep = random.randint(1, len(parent1[i]) - 1)

            weights = []

            childP1 = parent1[i][:sep]
            childP2 = parent2[i][sep:]

            for j in childP1:
                weights.append(j)

            for k in childP2:
                weights.append(k)

            child.append(weights)

            '''parentI = random.randint(1, 2)

            if parentI == 1:
                weights.append(parent1[i])
            if (parentI == 2):
                weights.append(parent2[i])

            child = weights'''

        return child

    def breedPopulation(self, matingpool, eliteSize, currentGeneration):
        children = []

        # length = len(matingpool) - eliteSize
        length = len(currentGeneration) - eliteSize

        for i in range(0, eliteSize):
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
            child = self.breed(parent1, parent2)
            children.append(child)

        return children

    def mutate(self, individual, mutationRate):
        for i in np.arange(len(individual)):
            # print('mutate weight', i)
            for neuron in range(len(individual[i])):
                if (random.random() < mutationRate):
                    newWeight = []
                    for j in range(len(individual[i][neuron])):
                        newWeight.append(random.uniform(-1., 1))
                        # print('neuronio', individual[i][neuron])

                    print('mutacao', newWeight)

                    individual[i][neuron] = newWeight

        return individual

    def mutatePopulation(self, population, mutationRate):
        mutatedPopulation = []

        for individual in range(0, len(population)):
            mutatedIndividual = self.mutate(population[individual], mutationRate)
            mutatedPopulation.append(mutatedIndividual)

        return mutatedPopulation

    def nextGeneration(self, currentGeneration, eliteSize, mutationRate):
        populationRanked = self.rankIndividuals(currentGeneration)

        selectionResults = self.selection(populationRanked, eliteSize)

        matingpool = self.matingPool(currentGeneration, selectionResults)

        children = self.breedPopulation(matingpool, eliteSize, currentGeneration)

        return self.mutatePopulation(children, mutationRate)


ga = GeneticAlgorithm(populationSize=150, eliteSize=2, mutationRate=0.05, generations=500)
ga.start()
