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


WIN = pygame.display.set_mode((800, 930))


# Colors 
BLACK = ((0, 0, 0))
WHITE = ((255, 255, 255))
GREEN = ((0, 255, 0))
BLUE = ((50, 50, 255))
RED = ((255, 0 , 0))

item_to_draw = ""


def predict(data, model, r):
    """
    Predicts what the drawing is

    Parameters: 
    data [dict]: Dictionary containing mapping between numbers and labels 
    model: Tensorflow model for doing predictions 
    r: Drawing window 
    """
    global item_to_draw

    # Get matrix of drawing area
    matrix = pygame.surfarray.array2d(WIN)
    matrix = matrix[:, 50:850]
    matrix = np.transpose(matrix)

    # Convert matrix to correspond to the training data
    image = np.expand_dims(matrix, axis=-1).astype(np.float32)
    image = abs((image - 16777215.) / 16777215.)
    image = cv2.resize(image, (28, 28))

    # Predict the drawing against the model and get 3 guesses
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

    # If guess is correct clear the screen and create a new item to draw
    if item_to_draw in [guess1, guess2, guess3]:
        print('Correct!')
        WIN.fill(WHITE, r)
        item_to_draw = list(data.values())[random.randint(0, len(data)-1)]

    else:
        print('Hmm incorrect, try again!')


def new_object(data, win, draw_window_rect, text_rect):
    """
    Clears screen and updates object to draw

    Parameters: 
    data [dict]: Dictionary containing mapping between numbers and labels 
    win: Main widow
    draw_window_rect: Drawing window
    text_rect: Rect for showing item to predict text
    """
    global item_to_draw

    # Choose a new random value and clear the screen
    item_to_draw = list(data.values())[random.randint(0, len(data)-1)]
    win.fill(WHITE, draw_window_rect)
    win.fill(WHITE, text_rect)


def main():
    global item_to_draw

    # Initialize pygame and font and load tensorflow model
    pygame.init()
    font = pygame.font.SysFont('Monospace', 30)
    model = keras.models.load_model("drawing_recognizer.h5")

    # For some reason the first predict is slow. Therefor we do it when the 
    # program starts
    model.predict(np.zeros((1, 28, 28, 1)))

    # Read the json file containing mapping between numbers and labels
    with open('num_to_image_dict.json', 'r') as f:
        data = json.load(f)

    # Rects for the drawing area and text area
    draw_window_rect = pygame.Rect(0, 50, 800, 800)
    text_rect = pygame.Rect(0, 0, 800, 30)

    WIN.fill((255, 255, 255))
    pygame.display.update()

    # Choose a random item to draw
    item_to_draw = list(data.values())[random.randint(0, len(data)-1)]

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            # Draw a circle around the mouse when mouse 1 is pressed, allow 
            # circles to be drawn inside the draw rect 
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[1] < 50 or mouse_pos[1] > 850:
                break
            pygame.draw.circle(WIN, (0, 0, 0), mouse_pos, 17)
            pygame.display.update()
            pygame.event.pump()

        while pygame.mouse.get_pressed()[2]:
            # Clear a circle around the mouse when mouse 2 is pressed
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[1] < 50 or mouse_pos[1] > 850:
                break
            pygame.draw.circle(WIN, (255, 255, 255),
                               mouse_pos, 17)
            pygame.display.update()
            pygame.event.pump()

        # Draw text to screen
        text = font.render(f'Item to draw: {item_to_draw}', True, (0, 0, 0))
        textRect = text.get_rect()
        WIN.blit(text, textRect)

        # Draw lines 
        upper_line = pygame.Rect(0, 30, 800, 20)
        lower_line = pygame.Rect(0, 850, 800, 20)
        pygame.draw.rect(WIN, BLACK, upper_line)
        pygame.draw.rect(WIN, BLACK, lower_line)

        # Add buttons for predicting, clearing and new item
        Button(WIN, 690, 875, 100, 50, text="Predict", inactiveColour=GREEN,
               onClick=lambda: predict(data, model, draw_window_rect))
        Button(WIN, 580, 875, 100, 50, text="Clear Image",
               onClick=lambda: WIN.fill(WHITE, draw_window_rect), inactiveColour=BLUE)
        Button(WIN, 470, 875, 100, 50, text="New Object", onClick=lambda: new_object(
            data, WIN, draw_window_rect, text_rect), inactiveColour=RED)

        pygame_widgets.update(events)
        pygame.display.update()


if __name__ == '__main__':
    main()
