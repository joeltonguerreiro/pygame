import numpy as np
import random
import math
import pygame
import sys
import time

from neural_network import NeuralNetwork
from ga import *

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

WIDTH, HEIGHT = 600, 300

SQUARE_WIDTH, SQUARE_HEIGHT = 8, 8
WALL_WIDTH, WALL_HEIGHT = 20, 20

velocity = 0.25
gravity = 0.1
velocity_jumping = 0.3
max_height_jumping = 70

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

font = pygame.font.Font('freesansbold.ttf', 12)

levels = [
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "B                            B",
    "B                            B",
    "B                            B",
    "B                            B",
    "B                            B",
    "B   PPPPPP                   B",
    "B                            B",
    "B          PPPPPPP           B",
    "B                            B",
    "B                  GGGGGG    B",
    "B             I              B",
    "BGGGGGGGGGGGGGGGGGGGGGGGGGGGGB",
    "B                            B",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
]

left = 0
top = 0

platforms = []
borders = []
grounds = []
exit = 0
pos_init = (0, 0)

#rasteira = pygame.image.load('rasteira.png')
#rasteira_resized = pygame.transform.scale(rasteira, (40, 40))
#rasteira_resized_b = pygame.transform.flip(rasteira_resized, True, False)

for row in levels:
    for col in row:
        if col == 'P':
            platform = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
            platforms.append(platform)

        if col == 'B':
            border = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
            borders.append(border)

        if col == 'E':
            exit = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)

        if col == 'I':
            pos_init = left, top

        if col == 'G':
            ground = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
            grounds.append(ground)

        left += WALL_WIDTH

    top += WALL_HEIGHT
    if left >= WIDTH:
        left = 0


class Square:
    def __init__(self, weights, rect, color):
        self.weights = weights
        self.rect = rect
        self.color = color
        self.jumping = True
        self.jumping_height = 1
        self.grabbed = False
        self.max_jumping_height = 80
        self.alive = True
        self.time_alive = 0
        self.jump_cooldown = 500
        self.last_jump_time = 0
        self.count_jumpings = 0
        self.velocity = 0.16

    def is_jumping(self):
        return self.jumping is True

    def is_grabbed(self):
        return self.grabbed is True

    def can_jump(self):
        now = int(round(time.time() * 1000))
        # print('can_jump', now - self.last_jump_time)

        if self.jumping is False and (now - self.last_jump_time) >= self.jump_cooldown:
            return True

        if self.jumping is True and self.jumping_height < self.max_jumping_height:
            return True

        return False

        '''if self.jumping_height < self.max_jumping_height:
            return True
        return False'''

    def jump(self):
        self.last_jump_time = int(round(time.time() * 1000))

    def increments_jump(self):
        self.count_jumpings += 1


neural_network = NeuralNetwork()


class Results:
    def __init__(self):
        self.winners = 0
        self.generations = 0

    def increment_winners(self, winner):
        self.winners += winner

    def reset_winners(self):
        self.winners = 0

    def increment_generations(self):
        self.generations += 1


results = Results()


class KillLine:
    def __init__(self, rect, color, velocity, direction):
        self.rect = rect
        self.color = color
        self.velocity = velocity
        self.direction = direction


class Fitness:
    def __init__(self, population):
        self.population = population

        self.list_squares = []

        self.fitness_results = {}
        self.has_squares_alive = True
        self.best_time = 0

        self.automatic_creation_kill_lasers = True
        self.list_kill_lines = []

    def create_kill_laser(self):
        kl_width = random.randint(3, 6)
        kl_height = random.randint(20, 60)
        kill_line_y = 260 - kl_height

        kill_line_direction = 'to_left'
        position_start_x = WIDTH
        if random.random() <= 0.1:
            kill_line_direction = 'to_right'
            position_start_x = 0

        kill_line_rect = pygame.Rect(position_start_x, kill_line_y, kl_width, kl_height)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        kill_line_velocity = random.uniform(0.18, 0.22)

        kill_line = KillLine(kill_line_rect, color, kill_line_velocity, kill_line_direction)
        self.list_kill_lines.append(kill_line)

    def run_game_rules(self):

        # results.resetWinners()

        # rasteira_rect = rasteira_resized.get_rect()
        # rasteira_rect.center = (0, 220)

        # rasteira_rect_b = rasteira_resized_b.get_rect()
        # rasteira_rect_b.center = (600, 220)

        start_time = int(round(time.time() * 1000))

        for i in range(0, len(self.population)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            square = Square(weights=self.population[i],
                            rect=pygame.Rect(pos_init[0], pos_init[1], SQUARE_WIDTH, SQUARE_HEIGHT),
                            color=color)

            self.list_squares.append(square)

        self.list_kill_lines = []

        while self.has_squares_alive:

            dt = clock.tick(60)

            pygame.display.set_caption('FPS: {}'.format(clock.get_fps()))

            #if rasteira_rect.centerx >= 600:
            #    rasteira_rect.centerx = 0

            #if rasteira_rect_b.centerx <= 0:
            #    rasteira_rect_b.centerx = 600

            screen.fill(BLACK)

            grab_position = 0, 0
            drag_position = 0, 0

            # eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # mouse button down
                if event.type == 5:
                    # print('event', event)
                    grab_position = event.pos

                    for square in self.list_squares:
                        if square.rect.collidepoint(grab_position[0], grab_position[1]):
                            square.grabbed = True
                            square.rect.inflate_ip(+3, +3)

                    drag_position = event.pos

                # mouse button up
                if event.type == 6:
                    # print('event', event)

                    for square in self.list_squares:
                        if square.is_grabbed():
                            square.rect.inflate_ip(-3, -3)

                            grab_position = 0, 0
                            square.grabbed = False
                            square.jumping = True
                            closer_ground_distance = 300
                            closer_ground = None

                # mouse motion
                if event.type == 4:
                    for square in self.list_squares:
                        if square.is_grabbed():
                            # print('event', event)
                            drag_position = event.pos

                if event.type == pygame.KEYUP:
                    # kill all
                    if event.key == pygame.K_SPACE:
                        for square in self.list_squares:
                            square.alive = False
                            die_time = int(round(time.time() * 1000))
                            square.time_alive = die_time - start_time
                            self.has_squares_alive = False
                        # continue

                    if event.key == pygame.K_a:
                        self.list_kill_lines = []

                        if self.automatic_creation_kill_lasers:
                            self.automatic_creation_kill_lasers = False

                        else:
                            self.automatic_creation_kill_lasers = True

                    if event.key == pygame.K_f:
                        self.create_kill_laser()

            if self.automatic_creation_kill_lasers and len(self.list_kill_lines) == 0:
                # chance de um laser ser criado
                kill_line_born_rate = random.random()

                if kill_line_born_rate <= 0.01:
                    self.create_kill_laser()



            for index in range(0, len(self.list_squares)):

                if not self.list_squares[index].alive:
                    continue

                current_distance = 300
                closer_ground_distance = 300
                closer_ground = None

                if self.list_squares[index].is_grabbed() is False:
                    # distancia até o chao / plataforma mais perto em cada sentido
                    distance_top = distance_bottom = distance_right = distance_left = 600

                    distance_enemies = {
                        'top': 600,
                        'bottom': 600,
                        'right': 600,
                        'left': 600
                    }

                    closer_enemies = {
                        'right': None,
                        'left': None
                    }

                    # objeto mais perto em cada sentido
                    closer_right = closer_bottom = closer_left = closer_top = None

                    for ground in grounds:

                        square = self.list_squares[index].rect

                        # o quadrado é menor que as paredes, então ele sempre estará
                        # contido em uma parede, os testes abaixo verificam esse caso

                        # if square.left >= ground.left and square.right <= ground.right:
                        if square.top >= ground.bottom:
                            if abs(square.top - ground.bottom) <= distance_top:
                                distance_top = abs(square.top - ground.bottom)
                                closer_top = ground

                        if square.bottom <= ground.top:
                            if abs(square.bottom - ground.top) <= distance_bottom:
                                distance_bottom = abs(square.bottom - ground.top)
                                closer_bottom = ground

                        # if square.top > ground.top and square.bottom < ground.bottom:
                        if square.right < ground.left:
                            if abs(square.right - ground.left) <= distance_right:
                                distance_right = abs(square.right - ground.left)
                                closer_right = ground

                        if square.left > ground.right:
                            if abs(square.left - ground.right) <= distance_left:
                                distance_left = abs(square.left - ground.right)
                                closer_left = ground

                    # distancia do laser mais perto
                    for kill_line in self.list_kill_lines:

                        square = self.list_squares[index].rect

                        # if square.top > kill_line.rect.top and square.bottom < kill_line.rect.bottom:
                        if square.right < kill_line.rect.left:
                            if abs(square.right - kill_line.rect.left) <= distance_enemies['right']:
                                distance_enemies['right'] = abs(square.right - kill_line.rect.left)
                                closer_enemies['right'] = kill_line.rect

                        if square.left > kill_line.rect.right:
                            if abs(square.left - kill_line.rect.right) <= distance_enemies['left']:
                                distance_enemies['left'] = abs(square.left - kill_line.rect.right)
                                closer_enemies['left'] = kill_line.rect

                    array_grounds = distance_right, distance_bottom, distance_left, distance_top

                    array_enemies = distance_enemies['right'], distance_enemies['left']

                    input_array = array_grounds + array_enemies

                    output = neural_network.feed_forward(self.list_squares[index].weights, input_array)

                    action = np.argmax(output)

                    # 0 direita
                    # 1 baixo
                    # 2 esquerda
                    # 3 pular

                    pos_x = pos_y = 0

                    if action == 0:
                        pos_x = pos_y = 0
                        closer_ground_distance = 300
                        closer_ground = None

                    if action == 1:
                        pos_x = (velocity * dt)
                        closer_ground_distance = 300
                        closer_ground = None

                    if action == 2:
                        pos_y = (velocity * dt)

                    if action == 3:
                        pos_x = -(velocity * dt)
                        closer_ground_distance = 300
                        closer_ground = None

                    if action == 4:
                        if self.list_squares[index].can_jump():
                            self.list_squares[index].jump()
                            pos_y = -(velocity_jumping * dt)
                            current_distance = 300
                            closer_ground_distance = 300
                            closer_ground = None
                            self.list_squares[index].jumping = True
                            self.list_squares[index].jumping_height += abs((velocity_jumping * dt))

                    for ground in grounds:

                        # se estiver caindo ou parado vai verificar o chao mais proximo
                        if pos_y >= 0:
                            current_distance = math.hypot(self.list_squares[index].rect.centerx - ground.centerx,
                                                          self.list_squares[index].rect.bottom - ground.top)

                            # se o chao testado for o mais proximo, atribui
                            if current_distance <= closer_ground_distance:
                                closer_ground = ground
                                closer_ground_distance = current_distance

                            if closer_ground is not None:
                                # testa se o quadrado saiu da plataforma
                                if pos_x > 0:
                                    if self.list_squares[index].rect.left < closer_ground.right and \
                                            self.list_squares[index].rect.bottom < closer_ground.top:
                                        self.list_squares[index].jumping = True

                                # testa se o quadrado saiu da plataforma
                                if pos_x < 0:
                                    if self.list_squares[index].rect.right > closer_ground.left and \
                                            self.list_squares[index].rect.bottom < closer_ground.top:
                                        self.list_squares[index].jumping = True

                                # verifica a colisao com o chao mais proximo
                                if self.list_squares[index].rect.colliderect(closer_ground):
                                    # print('colidiu')

                                    self.list_squares[index].jumping = False
                                    self.list_squares[index].jumping_height = 0

                                    self.list_squares[index].rect.bottom = closer_ground.top

                                    pos_y = 0

                            # se o estado for pulando aplica a gravidade
                            if self.list_squares[index].is_jumping():
                                # print('gravidade')
                                pos_y = (velocity + gravity) * dt

                        # se bateu na parte de baixo de uma plataforma
                        if self.list_squares[index].rect.colliderect(ground) and \
                                self.list_squares[index].rect.bottom > ground.bottom:
                            self.list_squares[index].jumping_height = max_height_jumping
                            self.list_squares[index].rect.top = ground.bottom
                            pos_y = 0

                    # not grabbed
                    if not self.list_squares[index].is_grabbed():
                        self.list_squares[index].rect.move_ip(pos_x, pos_y)

                    '''x1, y1 = self.list_squares[index].rect.center

                    if closer_right is not None:
                        r_x2, r_y2 = closer_right.center
                        pygame.draw.line(screen, GREEN, (x1, y1), (r_x2, r_y2), 1)

                    if closer_bottom is not None:
                        b_x2, b_y2 = closer_bottom.center
                        pygame.draw.line(screen, GREEN, (x1, y1), (b_x2, b_y2), 1)

                    if closer_left is not None:
                        l_x2, l_y2 = closer_left.center
                        pygame.draw.line(screen, GREEN, (x1, y1), (l_x2, l_y2), 1)

                    if closer_top is not None:
                        t_x2, t_y2 = closer_top.center
                        pygame.draw.line(screen, GREEN, (x1, y1), (t_x2, t_y2), 1)

                    if closer_enemies['right'] is not None:
                        r_x2, r_y2 = closer_enemies['right'].center
                        pygame.draw.line(screen, RED, (x1, y1), (r_x2, r_y2), 1)

                    if closer_enemies['left'] is not None:
                        l_x2, l_y2 = closer_enemies['left'].center
                        pygame.draw.line(screen, RED, (x1, y1), (l_x2, l_y2), 1)'''

                    # grabbed
                if self.list_squares[index].is_grabbed() and (drag_position[0] != 0 or drag_position[1] != 0):
                    self.list_squares[index].rect.center = drag_position

                for ground in grounds:
                    pygame.draw.rect(screen, WHITE, ground)

                for platform in platforms:
                    pygame.draw.rect(screen, WHITE, platform)

                for border in borders:
                    pygame.draw.rect(screen, RED, border)

                    if self.list_squares[index].rect.colliderect(border):
                        # print('bateu')
                        if pos_x > 0:
                            self.list_squares[index].rect.right = border.left

                        if pos_x < 0:
                            self.list_squares[index].rect.left = border.right

                # pygame.draw.rect(screen, GREEN, exit)

                # colisao com os lasers
                for kill_line in self.list_kill_lines:
                    if self.list_squares[index].rect.colliderect(kill_line.rect):
                        self.list_squares[index].alive = False

                # tempo de vida do individuo
                if self.list_squares[index].alive is False:
                    die_time = int(round(time.time() * 1000))
                    self.list_squares[index].time_alive = die_time - start_time

                # results.incrementWinners(1)

                pygame.draw.rect(screen, self.list_squares[index].color, self.list_squares[index].rect)

            for kill_line in self.list_kill_lines:
                kl_pos_x = dt * kill_line.velocity
                if kill_line.direction == 'to_left':
                    kl_pos_x = kl_pos_x * -1

                kill_line.rect.move_ip(kl_pos_x, 0)
                pygame.draw.rect(screen, kill_line.color, kill_line.rect)

                if (kill_line.direction == 'to_left' and kill_line.rect.left <= 0) or (kill_line.direction == 'to_right' and kill_line.rect.right >= WIDTH):
                    self.list_kill_lines = []

            alives = 0

            for square in self.list_squares:
                if square.alive:
                    alives += 1

            if alives == 0:
                self.has_squares_alive = False

            generations_text = 'geracao: ' + str(results.generations)
            generations_font = font.render(generations_text, True, WHITE)
            generations_rect = generations_font.get_rect()
            generations_rect.center = (80, 60)

            screen.blit(generations_font, generations_rect)

            alives_text = 'vivos: ' + str(alives)
            alives_font = font.render(alives_text, True, WHITE)
            alives_rect = alives_font.get_rect()
            alives_rect.center = (80, 80)

            screen.blit(alives_font, alives_rect)

            # winners_text = 'vencedores: ' + str(results.getWinners())
            # winners_font = font.render(winners_text, True, WHITE)
            # winners_rect = winners_font.get_rect()
            # winners_rect.center = (50, HEIGHT - 40)

            # screen.blit(winners_font, winners_rect)

            # rasteira_rect.move_ip(dt * kill_line_velocity, 0)
            # screen.blit(rasteira_resized, rasteira_rect)

            # rasteira_rect_b.move_ip(-(dt * kill_line_velocity), 0)
            # screen.blit(rasteira_resized_b, rasteira_rect_b)

            pygame.display.flip()

        return True

    def get_fitness(self):

        self.run_game_rules()

        # print('bestDistance', self.bestDistance)

        for index in range(0, len(self.list_squares)):
            time_alive = self.list_squares[index].time_alive

            #count_jumpings = self.list_squares[index].count_jumpings

            fitness = 1

            if time_alive > 0:
                fitness = 1 / time_alive

            # print('fitness', fitness)

            self.fitness_results[index] = fitness

        return self.fitness_results


class GeneticAlgorithm:
    def __init__(self, population_size, elite_size, mutation_rate, generations, fitness):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.current_population = []
        self.fitness = fitness

    def start(self):
        input_size = 6
        hidden_size = 10
        output_size = 5  # parado, pular, direita, esquerda, baixo
        self.current_population = initial_population(self.population_size, input_size, hidden_size, output_size)

        for i in range(0, self.generations):
            print('generation: ', i)
            results.increment_generations()
            self.current_population = next_generation(self.current_population,
                                                      self.elite_size,
                                                      self.mutation_rate,
                                                      self.fitness
                                                      )


ga = GeneticAlgorithm(population_size=50, elite_size=0, mutation_rate=0.02, generations=500, fitness=Fitness)
ga.start()
