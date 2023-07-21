import pygame
from tensorflow import keras
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import time
import copy


WIN = pygame.display.set_mode((800,900))


def main():
    pygame.init()
    font = pygame.font.SysFont('Arial', 30)
    model = keras.models.load_model("drawing_recognizer.h5")
    model.predict(np.zeros((1, 28, 28, 1))) # For some reason the first predict is slow. 
                                            # So this is run so all future predicts are quick
    # Read the json file
    with open('num_to_image_dict.json', 'r') as f:
        data = json.load(f)

    WIN.fill((255, 255, 255))
    pygame.display.update()
    item_to_draw = list(data.values())[random.randint(0,len(data)-1)]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            # Draw a rect around the mouse
            pygame.draw.circle(WIN, (0, 0, 0), pygame.mouse.get_pos(), 17)
            pygame.display.update()
            pygame.event.pump()

        while pygame.mouse.get_pressed()[2]:
            pygame.draw.circle(WIN, (255, 255, 255), pygame.mouse.get_pos(), 17)
            pygame.display.update()
            pygame.event.pump()

        if pygame.key.get_pressed()[pygame.K_SPACE]:
            WIN.fill((255, 255, 255))

        if pygame.key.get_pressed()[pygame.K_r]:
            item_to_draw = list(data.values())[random.randint(0,len(data)-1)]
            WIN.fill((255, 255, 255))

        if pygame.key.get_pressed()[pygame.K_a]:
            matrix = pygame.surfarray.array2d(WIN)
            matrix = matrix[:, 100:]
            matrix = np.transpose(matrix)
            image = np.expand_dims(matrix, axis=-1).astype(np.float32)
            image = abs((image - 16777215.) / 16777215.)

            image = cv2.resize(image, (28, 28))
            plt.imshow(image)
            plt.show()

            ans = model.predict(image.reshape(1, 28, 28, 1))
            data_copy = copy.deepcopy(data)
            guess = data_copy[str(ans.argmax())]
            data_copy.pop(guess)
            print(f'My guess is: {guess}')
            guess = data_copy[str(ans.argmax())]
            data_copy.pop(guess)
            print(f'My guess is: {guess}')
            guess = data_copy[str(ans.argmax())]
            data_copy.pop(guess)
            print(f'My guess is: {guess}')
            if guess == item_to_draw: 
                print('Correct!')
                WIN.fill((255, 255, 255))
                item_to_draw = list(data.values())[random.randint(0,len(data)-1)]

            else:
                print('Hmm incorrect, try again!')
        text = font.render(f'Item to draw: {item_to_draw}', True, (0, 0, 0))
        textRect = text.get_rect()
        WIN.blit(text, textRect)
        pygame.display.update()
        


if __name__ == '__main__':
    main()