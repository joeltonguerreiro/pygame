import numpy as np
import operator
import pandas as pd
import random
import math
import pygame
import sys

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
WIDTH, HEIGHT = 600, 300

SQUARE_WIDTH, SQUARE_HEIGHT = 8, 8
WALL_WIDTH, WALL_HEIGHT = 20, 20

velocity = 0.16
gravity = 0.05
velocity_jumping = 0.14
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
    "B        E    GGG            B",
    "B                            B",
    "B                   GG       B",
    "B                            B",
    "B             GGGGGG         B",
    "B     I                      B",
    "BGGGGGGGGGGGGGGGGGGGGGGGGGGGGB",
    "B                            B",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
]

left = 0
top = 0

walls = []
borders = []
grounds = []
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

        if col == 'I':
            pos_init = left, top

        if col == 'G':
            ground = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
            grounds.append(ground)

        left += WALL_WIDTH

    top += WALL_HEIGHT
    if left >= WIDTH:
        left = 0

square = [pygame.Rect(pos_init[0], pos_init[1], SQUARE_WIDTH, SQUARE_HEIGHT), 1, max_height_jumping, 0]
# rect
# jumping 1, 0
# jumping_height
# grabbed

currentDistance = 300
closerGroundDistance = 300
closerGround = None
grab_position = 0, 0
drag_position = 0, 0

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # mouse button down
        if event.type == 5:
            print('event', event)
            grab_position = event.pos

            if square[0].collidepoint(grab_position[0], grab_position[1]):
                square[3] = 1
                square[0].inflate_ip(+3, +3)

            drag_position = event.pos

        # mouse button up
        if event.type == 6:
            print('event', event)

            if square[3] == 1:
                square[0].inflate_ip(-3, -3)

            grab_position = 0, 0
            square[3] = 0
            square[1] = 1
            closerGroundDistance = 300
            closerGround = None

        # mouse motion
        if event.type == 4 and square[3] == 1:
            print('event', event)
            drag_position = event.pos

    keys = pygame.key.get_pressed()

    velocity_x = velocity_y = 0

    dt = clock.tick(60)

    if keys[pygame.K_UP]:
        if square[2] < max_height_jumping:
            velocity_y = -(velocity_jumping * dt)
            currentDistance = 300
            closerGroundDistance = 300
            closerGround = None
            square[1] = 1
            square[2] += abs((velocity_jumping * dt))

    if keys[pygame.K_DOWN]:
        velocity_y = (velocity * dt)

    if keys[pygame.K_LEFT]:
        velocity_x = -(velocity * dt)
        closerGroundDistance = 300
        closerGround = None

    if keys[pygame.K_RIGHT]:
        velocity_x = (velocity * dt)
        closerGroundDistance = 300
        closerGround = None

    for ground in grounds:

        # se estiver caindo ou parado vai verificar o chao mais proximo
        if velocity_y >= 0:
            currentDistance = math.hypot(square[0].centerx - ground.centerx, square[0].bottom - ground.top)

            # se o chao testado for o mais proximo, atribui
            if currentDistance <= closerGroundDistance:
                closerGround = ground
                closerGroundDistance = currentDistance

            # testa se o quadrado saiu da plataforma
            if velocity_x > 0:
                if square[0].left < closerGround.right and square[0].bottom <= closerGround.top:
                    square[1] = 1

            # testa se o quadrado saiu da plataforma
            if velocity_x < 0:
                if square[0].right > closerGround.left and square[0].bottom <= closerGround.top:
                    square[1] = 1

            # verifica a colisao com o chao mais proximo
            if square[0].colliderect(closerGround):
                print('colidiu')

                square[1] = 0
                square[2] = 0

                square[0].bottom = closerGround.top

                velocity_y = 0

            # se o estado for 1 (pulando) aplica a gravidade
            if square[1] == 1:
                print('gravidade')
                velocity_y = (velocity + gravity) * dt

    for ground in grounds:
        if square[0].colliderect(ground) and square[0].bottom > ground.bottom:
            square[2] = max_height_jumping
            square[0].top = ground.bottom
            velocity_y = 0

    # not dragged
    if square[3] == 0:
        square[0].move_ip(velocity_x, velocity_y)

    # dragged
    if square[3] == 1:
        square[0].center = drag_position

    screen.fill(BLACK)

    if closerGround is not None:
        pygame.draw.line(screen, RED, (square[0].centerx, square[0].centery),
                         (closerGround.centerx, closerGround.centery), 1)

    for border in borders:
        pygame.draw.rect(screen, RED, border)

    for ground in grounds:
        pygame.draw.rect(screen, WHITE, ground)

    pygame.draw.rect(screen, GREEN, exit)

    pygame.draw.rect(screen, GREEN, square[0])

    pygame.display.flip()
