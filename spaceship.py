import numpy as np
import random
import math
import pygame
import sys
import time

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

WIDTH, HEIGHT = 600, 300

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

while True:
    dt = clock.tick(60)

    screen.fill(BLACK)

    pygame.display.flip()
