import pygame
from tensorflow import keras
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import random


WIN = pygame.display.set_mode((800,800))


def main():
    model = keras.models.load_model("drawing_recognizer.h5")
    model.predict(np.zeros((1, 28, 28, 1))) # For some reason the first predict is slow. 
                                            # So this is run so all future predicts are quick
    # Read the json file
    with open('num_to_image_dict.json', 'r') as f:
        data = json.load(f)

    WIN.fill((255, 255, 255))
    pygame.display.update()
    item_to_draw = list(data.values())[random.randint(0,len(data)-1)]
    print(f'Item to draw: {item_to_draw}')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            # Draw a rect around the mouse
            r = pygame.Rect(pygame.mouse.get_pos(), (10, 10))
            pygame.draw.circle(WIN, (0, 0, 0), pygame.mouse.get_pos(), 13)
            pygame.display.update()
            pygame.event.pump()

        while pygame.mouse.get_pressed()[2]:
            pygame.draw.circle(WIN, (255, 255, 255), pygame.mouse.get_pos(), 13)
            pygame.display.update()
            pygame.event.pump()

        if pygame.key.get_pressed()[pygame.K_SPACE]:
            WIN.fill((255, 255, 255))
            pygame.display.update()

        if pygame.key.get_pressed()[pygame.K_a]:
            matrix = pygame.surfarray.array2d(WIN)
            matrix = np.transpose(matrix)
            image = np.expand_dims(matrix, axis=-1).astype(np.float32)
            image = abs((image - 16777215.) / 16777215.)

            image = cv2.resize(image, (28, 28))

            ans = model.predict(image.reshape(1, 28, 28, 1))
            guess = data[str(ans.argmax())]
            print(f'My guess is: {guess}')
            if guess == item_to_draw: 
                print('Correct, you win!')
                quit()
            else:
                print('Hmm incorrect, try again!')
        


if __name__ == '__main__':
    main()