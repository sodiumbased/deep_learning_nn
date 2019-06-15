'''
Authors
Alexander Kung, Stephen Wang
'''

import pygame
import sys
import numpy as np
from Layer import Layer

pygame.init()
screen_size = (560,590)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Draw your digit')

import json
with open('parameters/last_epoch.json', 'r') as f:
    js_obj = json.load(f)

network = []
network.append(Layer(theta=np.reshape(js_obj['theta1'], (50,785))))
network.append(Layer(theta=np.reshape(js_obj['theta2'], (50,51))))
network.append(Layer(theta=np.reshape(js_obj['theta3'], (10,51))))
network.append(Layer())

radius = 30

drawingArea = np.zeros((560,560))


def brightness_assign(canvas, mouse_pos):
    a = -255/radius**2
    for i in range(mouse_pos[0]-radius, mouse_pos[0]+radius+1):
        for b in range(mouse_pos[1]-radius, mouse_pos[1]+radius+1):
            canvas[i][b] = 255
            # distance = ((i-mouse_pos[0])**2 + (b-mouse_pos[1])**2)**.5
            # if distance > radius:
            #     continue
            # else:
            #     brightness_value = int(a * distance ** 2 + 255)
            #     try:
            #         if brightness_value > canvas[i][b]:
            #             canvas[i][b] = brightness_value
            #         else:
            #             continue
            #     except:
            #         continue

def brightness_assign2(sf, mouse_pos):
    a = -255/radius**2
    for i in range(mouse_pos[0]-radius, mouse_pos[0]+radius+1):
        for b in range(mouse_pos[1]-radius, mouse_pos[1]+radius+1):
            #sf.set_at((i,b),(255,255,255))
            distance = ((i-mouse_pos[0])**2 + (b-mouse_pos[1])**2)**.5
            if distance > radius:
                continue
            else:
                brightness_value = int(a * distance ** 2 + 255)
                try:
                    if brightness_value > tuple(sf.get_at((i,b)))[0]:
                        sf.set_at((i,b),(brightness_value,brightness_value,brightness_value))
                    else:
                        continue
                except:
                    continue

def nn_evaluate(canvas):
    global network
    network[0].a = canvas
    network[1].activate(network[0],next_to_input=True)
    for i in range(2,len(network)):
        network[i].activate(network[i-1])
    temp = list(np.reshape(network[3].a, (10)))
    print(temp)
    return temp.index(max(temp))

def downscale(canvas):
    temp = np.zeros((28,28))
    for i in range(28):
        for b in range(28):
            #print(canvas[i*20:(i+1)*20][b*20:(b+1)*20].sum())
            temp[i][b] = canvas[i*20:(i+1)*20, b*20:(b+1)*20].sum()/400
    
    return np.reshape(np.insert(np.reshape(temp, (784)), 0, 1), (1,785))

sf = pygame.Surface ((560,560))
sf.fill((0,0,0))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    if pygame.mouse.get_pressed()[0] == True:
        position = pygame.mouse.get_pos()
        if position[1] <= 560:
            brightness_assign2(sf, position)
        elif position[1] > 560:
            if position[0] > 100:
                for i in range(drawingArea.shape[0]):
                    for j in range(drawingArea.shape[1]):
                        drawingArea[j][i] = tuple(sf.get_at((i,j)))[0]

                import matplotlib.pyplot as plt
                pixels = np.reshape(downscale(drawingArea)[0,1:], (28,28))
                plt.imshow(pixels, cmap='gray')
                plt.show()
                print(nn_evaluate(downscale(drawingArea)))
            else:
                sf.fill((0,0,0))
                drawingArea = np.zeros((560,560))
            
    pygame.draw.rect(screen, [0, 255, 0], [100, 560, 460, 30], 0)
    pygame.draw.rect(screen, [255, 0, 0], [0, 560, 100, 30], 0)
    screen.blit(sf, (0, 0))

    pygame.display.flip()