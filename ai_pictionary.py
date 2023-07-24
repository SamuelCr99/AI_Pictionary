import pygame
from tensorflow import keras
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import time
import copy
import pygame_widgets
from pygame_widgets.button import Button


WIN = pygame.display.set_mode((800,930))


# Colors
BLACK = ((0,0,0))
WHITE = ((255,255,255))
GREEN = ((0, 255, 0))

item_to_draw = ""

def predict(data, model, r):
    global item_to_draw


    matrix = pygame.surfarray.array2d(WIN)
    matrix = matrix[:, 50:850]
    matrix = np.transpose(matrix)
    image = np.expand_dims(matrix, axis=-1).astype(np.float32)
    image = abs((image - 16777215.) / 16777215.)

    image = cv2.resize(image, (28, 28))

    ans_arr = model.predict(image.reshape(1, 28, 28, 1))
    ans_copy = copy.deepcopy(ans_arr)
    guess1 = data[str(ans_copy.argmax())]
    ans_copy = np.delete(ans_copy, ans_copy.argmax())
    guess2 = data[str(ans_copy.argmax())]
    ans_copy = np.delete(ans_copy, ans_copy.argmax())
    guess3 = data[str(ans_copy.argmax())]

    print(f'My first guess is: {guess1}')
    print(f'My second guess is: {guess2}')
    print(f'My third guess is: {guess3}')

    if item_to_draw in [guess1, guess2, guess3]: 
        print('Correct!')
        WIN.fill(WIN , r)
        item_to_draw = list(data.values())[random.randint(0,len(data)-1)]

    else:
        print('Hmm incorrect, try again!')

def new_object(data, win, draw_window_rect, text_rect):
    global item_to_draw

    item_to_draw = list(data.values())[random.randint(0,len(data)-1)]
    win.fill(WHITE, draw_window_rect)
    win.fill(WHITE, text_rect)


def main():
    global item_to_draw
    pygame.init()
    font = pygame.font.SysFont('Arial', 30)
    model = keras.models.load_model("drawing_recognizer.h5")
    model.predict(np.zeros((1, 28, 28, 1))) # For some reason the first predict is slow. 
                                            # So this is run so all future predicts are quick
    # Read the json file
    with open('num_to_image_dict.json', 'r') as f:
        data = json.load(f)

    draw_window_rect = pygame.Rect(0, 50, 800, 800)
    text_rect = pygame.Rect(0,0, 800, 30 )

    WIN.fill((255, 255, 255))
    pygame.display.update()
    item_to_draw = list(data.values())[random.randint(0,len(data)-1)]

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            # Draw a rect around the mouse
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[1] < 50 or mouse_pos[1] > 850:
                break
            pygame.draw.circle(WIN, (0, 0, 0), mouse_pos, 17)
            pygame.display.update()
            pygame.event.pump()

        while pygame.mouse.get_pressed()[2]:
            pygame.draw.circle(WIN, (255, 255, 255), pygame.mouse.get_pos(), 17)
            pygame.display.update()
            pygame.event.pump()

        text = font.render(f'Item to draw: {item_to_draw}', True, (0, 0, 0))
        textRect = text.get_rect()
        WIN.blit(text, textRect)
        upper_line = pygame.Rect(0, 30, 800, 20)
        lower_line = pygame.Rect(0, 850, 800, 20)
        pygame.draw.rect(WIN, BLACK, upper_line)
        pygame.draw.rect(WIN, BLACK, lower_line)

        Button(WIN, 690, 875, 100, 50, text="Predict", inactiveColour = GREEN, onClick = lambda: predict(data, model, draw_window_rect)) 
        Button(WIN, 580, 875, 100, 50, text="Clear Image", onClick = lambda: WIN.fill(WHITE, draw_window_rect)) 
        Button(WIN, 470, 875, 100, 50, text="New Object", onClick = lambda: new_object(data, WIN, draw_window_rect, text_rect)) 

        pygame_widgets.update(events)  
        pygame.display.update()
        


if __name__ == '__main__':
    main()