import pygame

pygame.init()

WINDOW_SIZE = WIDTH, HEIGHT = 1000 // 2, 750 // 2

screen = pygame.display.set_mode(WINDOW_SIZE)

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)


class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image_file).convert()
        self.resized = pygame.transform.scale(self.image, WINDOW_SIZE)
        self.rect = self.resized.get_rect()
        self.rect.left, self.rect.top = position


background = Background('freetileset/png/BG/BG.png', [0, 0])


GROUND_FILES = [
    'freetileset/png/Tiles/1.png',
    'freetileset/png/Tiles/2.png',
    'freetileset/png/Tiles/3.png',
    'freetileset/png/Tiles/4.png',
    'freetileset/png/Tiles/5.png',
    'freetileset/png/Tiles/6.png',
]


class Ground(pygame.sprite.Sprite):
    def __init__(self, image_file, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image_file)
        self.resized = pygame.transform.scale(self.image, (WIDTH // 20, HEIGHT // 20))
        self.rect = self.resized.get_rect()
        self.rect.left, self.rect.top = position
        self.ground = None


list_grounds = []

left, top = WIDTH // 10 - 40, HEIGHT - (HEIGHT // 10)

for i in range(len(GROUND_FILES)):
    ground = Ground(GROUND_FILES[i], [left, top])

    left += WIDTH // 20

    if i == 2:
        left = WIDTH // 10 - 40
        top += HEIGHT // 20

    list_grounds.append(ground)

left, top = WIDTH // 5 - 20, HEIGHT - (HEIGHT // 10)

for i in range(len(GROUND_FILES)):
    ground = Ground(GROUND_FILES[i], [left, top])

    left += WIDTH // 20

    if i == 2:
        left = WIDTH // 5 - 20
        top += HEIGHT // 20

    list_grounds.append(ground)

while True:
    screen.fill(WHITE)

    screen.blit(background.resized, background.rect)

    for ground in list_grounds:
        screen.blit(ground.resized, ground.rect)

    pygame.display.flip()
