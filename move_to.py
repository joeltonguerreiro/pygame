import math
import sys

import pygame
import pygame_gui

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)

WIDTH, HEIGHT = 600, 300

SQUARE_WIDTH, SQUARE_HEIGHT = 20, 20

WALL_WIDTH, WALL_HEIGHT = 20, 20

pygame.init()

window = (WIDTH, HEIGHT)

screen = pygame.display.set_mode(window)
manager = pygame_gui.UIManager(window)
background = pygame.Surface(window)

hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((500, 250), (100, 50)),
                                             text='reset',
                                             manager=manager)

clock = pygame.time.Clock()

square = pygame.Rect(WIDTH // 2, HEIGHT // 2, SQUARE_WIDTH, SQUARE_HEIGHT)

velocity = 0.1

destiny = 0, 0

pos_x = pos_y = 0

a = b = x1 = x2 = y1 = y2 = 0

font = pygame.font.Font('freesansbold.ttf', 12)

levels = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "W             WW             W",
    "W    WWWWW    WW             W",
    "W             WW             W",
    "W    WWWWW    WW             W",
    "W             WW             W",
    "W                            W",
    "W                   WWWWW    W",
    "W                            W",
    "W             WW             W",
    "W             WW             W",
    "W                            W",
    "W                            W",
    "W                            W",
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"
]

left = 0
top = 0

walls = []

for row in levels:
    for col in row:
        if col == 'W':
            wall = pygame.Rect(left, top, WALL_WIDTH, WALL_HEIGHT)
            walls.append(wall)
        left += WALL_WIDTH
    top += WALL_HEIGHT
    if left >= WIDTH:
        left = 0

pause = False

# background.fill(BLACK)

for wall in walls:
    pygame.draw.rect(background, WHITE, wall)

screen.blit(background, (0, 0))

pygame.display.update()

while True:

    dt = clock.tick(90)

    events = pygame.event.get()

    for event in events:

        if pygame.event.event_name(event.type) == 'MouseButtonDown':
            destiny = event.pos

        # if event.type == pygame.KEYUP:
        #     if event.key == pygame.K_SPACE:
        #         square.centerx = WIDTH // 2
        #         square.centery = HEIGHT // 2
        #         destiny = 0, 0
        #         continue

        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == hello_button:
                    print('Hello World!')
                    square.centerx = WIDTH // 2
                    square.centery = HEIGHT // 2
                    destiny = 0, 0
                    continue

        manager.process_events(event)

    manager.update(dt)

    if destiny[0] != 0 or destiny[1] != 0:
        # verifica se a posição destino é diferente de 0
        x1 = square.centerx
        y1 = square.centery
        x2 = destiny[0]
        y2 = destiny[1]

        screen.blit(background, (0, 0))
        new = pygame.draw.line(background, RED, (x1, y1), (x2, y2), 1)
        screen.blit(background, (0, 0))
        pygame.display.update(new)

        try:
            a = b = 0

            if (x2 - x1) != 0:
                # calcula a distância entre as posições de origem e destino
                a = (y2 - y1) / (x2 - x1)
                b = ((y1 * x2) - (y2 * x1)) / (x2 - x1)

                x = square.centerx
                y = (a * x) + b
            else:
                print('divisao por 0', x2, x1)
                print('y2:', y2, 'y1:', y1)
                x = 0
                y = square.centery
                # continue

        except Exception as e:
            print('except', e)
            pygame.quit()
            sys.exit()



        # verifica se chegou
        if square.centerx == destiny[0] and square.centery == destiny[1]:
            print('chegou')
            destiny = 0, 0

        pos_x = 0
        if destiny[0] != x:
            pos_x = math.floor(velocity * dt)
            if destiny[0] < x:
                pos_x = -pos_x

        pos_y = 0
        if destiny[1] != y:
            pos_y = math.floor(velocity * dt)

            if destiny[1] < y:
                pos_y = -pos_y

        dirty_rects = [square]

        screen.blit(background, square)

        # verifica se deve andar
        if pos_x != 0 or pos_x != x2:
            square.move_ip(pos_x, 0)

        for wall in walls:
            if square.colliderect(wall):
                if pos_x > 0:
                    square.right = wall.left

                if pos_x < 0:
                    square.left = wall.right

        if pos_y != 0 or pos_y != y2:
            square.move_ip(0, pos_y)

        # dirty_rects.append(square)

        pygame.draw.rect(background, RED, square)

        # screen.blit(background, square)

        pygame.display.update(dirty_rects)

        for wall in walls:
            if square.colliderect(wall):
                print('colidiu y', pos_y)
                if pos_y > 0:
                    square.bottom = wall.top

                if pos_y < 0:
                    square.top = wall.bottom

    str_pos_square = 'square x=' + str(square.centerx) + ', y=' + str(square.centery)
    text_square = font.render(str_pos_square, True, WHITE)
    text_rect_square = text_square.get_rect()
    text_rect_square.center = (WIDTH // 2, HEIGHT - 60)

    screen.blit(text_square, text_rect_square)

    str_destiny = 'destiny x=' + str(destiny[0]) + ', y=' + str(destiny[1])
    text_destiny = font.render(str_destiny, True, WHITE)
    text_rect_destiny = text_destiny.get_rect()
    text_rect_destiny.center = (WIDTH // 2, HEIGHT - 40)

    screen.blit(text_destiny, text_rect_destiny)

    manager.draw_ui(screen)

    # pygame.display.flip()



