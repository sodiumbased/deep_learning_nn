import pygame
import sys
import numpy as np

pygame.init()
screen_size = (560,590)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Draw your digit')

drawingArea = np.array([])

radius = 56

drawingArea = np.zeros((560,560))

class Button(pygame.sprite.Sprite):
    """Class used to create a button, use setCords to set 
        position of topleft corner. Method pressed() returns
        a boolean and should be called inside the input loop."""
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image, self.rect = load_image('button.png', -1)

    def setCords(self,x,y):
        self.rect.topleft = x,y

    def pressed(self,mouse):
        if mouse[0] > self.rect.topleft[0]:
            if mouse[1] > self.rect.topleft[1]:
                if mouse[0] < self.rect.bottomright[0]:
                    if mouse[1] < self.rect.bottomright[1]:
                        return True
                    else: return False
                else: return False
            else: return False
        else: return False

def brightness_assign(canvas, mouse_pos):
    a = -255/radius**2
    for i in range(mouse_pos[0]-radius, mouse_pos[0]+radius+1):
        for a in range(mouse_pos[1]-radius, mouse_pos[0]+radius+1):
            distance = (i**2 + a**2)**.5
            try:
                canvas[i][a] = a * distance ** 2 + 255
            except:
                pass

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    
    if pygame.mouse.get_pressed()[0] == True:
        position = pygame.mouse.get_pos()
        if position[1] <= 560:
            brightness_assign(drawingArea, position)

    
    pygame.fill((0,0,0))

    background = pygame.Rect()


    pygame.display.flip()