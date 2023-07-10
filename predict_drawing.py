import pygame
from tensorflow import keras
import cv2
import numpy as np
import json


WIN = pygame.display.set_mode((800,800))

def translate_colors(colors):
    for i in range(len(colors)):
        for j in range(len(colors[i])):
            if colors[i][j] == 16777215:
                colors[i][j] = 1
            else:
                colors[i][j] = 0
    return colors

def find_key_from_value(d, value):
    for k, v in d.items():
        if v == value:
            return k

def main():
    model = keras.models.load_model("drawing_recognizer.h5")

    with open('labels.json', 'r') as f:
        labels = json.load(f)

    WIN.fill((255, 255, 255))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            pygame.draw.circle(WIN, (0, 0, 0), pygame.mouse.get_pos(), 4)
            pygame.display.update()
            pygame.event.pump()

        while pygame.mouse.get_pressed()[2]:
            pygame.draw.circle(WIN, (255, 255, 255), pygame.mouse.get_pos(), 4)
            pygame.display.update()
            pygame.event.pump()

        if pygame.key.get_pressed()[pygame.K_SPACE]:
            WIN.fill((255, 255, 255))
            pygame.display.update()

        if pygame.key.get_pressed()[pygame.K_a]:
            matrix = pygame.surfarray.array2d(WIN)
            correct_matrix = translate_colors(matrix).transpose()
            print(correct_matrix)
            image = np.expand_dims(correct_matrix, axis=-1).astype(np.float32)
            image = cv2.resize(image, (100, 100))
            ans = model.predict(image.reshape(1, 100, 100, 1))
            print(find_key_from_value(labels, ans.argmax()))
        


if __name__ == '__main__':
    main()