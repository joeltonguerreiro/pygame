import pygame
import pygame_gui

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
GRAY = pygame.Color(20, 20, 20)

WIDTH, HEIGHT = 700, 300
window = (WIDTH, HEIGHT)
game_window = (WIDTH - 100, HEIGHT)
ui_window = (100, HEIGHT)

SQUARE_WIDTH, SQUARE_HEIGHT = 5, 5

destiny = 0, 0

pygame.init()

screen = pygame.display.set_mode(window)
manager = pygame_gui.UIManager(window)

clock = pygame.time.Clock()

square = pygame.Rect(0, 0, SQUARE_WIDTH, SQUARE_HEIGHT)


def draw_ui_window():
    background2 = pygame.Surface(ui_window)
    background2.fill(GRAY)
    dirty.append(pygame.draw.circle(background2, GREEN, (10, 10), 5))
    dirty.append(screen.blit(background2, (WIDTH - 100, 0)))

    return dirty


def draw_game_window(new_square):
    background = pygame.Surface(game_window)
    background.fill(BLACK)
    dirty.append(pygame.draw.rect(background, WHITE, new_square))
    dirty.append(screen.blit(background, (0, 0)))

    return dirty


while True:
    dt = clock.tick(20)

    pygame.display.set_caption('FPS: {}'.format(clock.get_fps()))

    dirty = [screen.fill(BLACK)]

    events = pygame.event.get()
    for event in events:
        manager.process_events(event)

        if pygame.event.event_name(event.type) == 'MouseButtonDown':
            destiny = event.pos
            print(destiny)

    if destiny[0] != 0 or destiny[1] != 0:
        square.center = destiny

    dirty = dirty + draw_game_window(square)
    dirty = dirty + draw_ui_window()

    pygame.display.update(dirty)
