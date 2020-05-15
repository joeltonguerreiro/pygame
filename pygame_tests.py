import pygame
import pygame_gui

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

WIDTH, HEIGHT = 600, 300
window = (WIDTH, HEIGHT)

SQUARE_WIDTH, SQUARE_HEIGHT = 5, 5

destiny = 0, 0

pygame.init()

screen = pygame.display.set_mode(window)
manager = pygame_gui.UIManager(window)

clock = pygame.time.Clock()

square = pygame.Rect(0, 0, SQUARE_WIDTH, SQUARE_HEIGHT)

while True:
    dt = clock.tick(20)

    pygame.display.set_caption('FPS: {}'.format(clock.get_fps()))

    dirty = [screen.fill(BLACK)]

    background = pygame.Surface((100, 100))
    background.fill(RED)
    dirty.append(screen.blit(background, (0, 0)))

    background2 = pygame.Surface((100, 100))
    background2.fill(GREEN)
    dirty.append(screen.blit(background2, (0, 120)))

    events = pygame.event.get()
    for event in events:
        manager.process_events(event)

        if pygame.event.event_name(event.type) == 'MouseButtonDown':
            destiny = event.pos
            print(destiny)

    if destiny[0] != 0 or destiny[1] != 0:
        square.center = destiny

        dirty.append(pygame.draw.rect(screen, WHITE, square))

    pygame.display.update(dirty)

