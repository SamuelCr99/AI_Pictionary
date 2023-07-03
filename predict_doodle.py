import pygame
from tensorflow import keras
import cv2
import numpy as np


WIN = pygame.display.set_mode((256, 256))
x_y_values = []
x_values = []
y_values = []

def translate_array(colors):
    for i in range(len(colors)):
        for j in range(len(colors[i])):
            if not colors[i][j] == 16777215 and [i, j] not in x_y_values:
                x_y_values.append([i, j])
                x_values.append(i)
                y_values.append(j)

def main():
    model = keras.models.load_model("digit-recognizer.h5")
    WIN.fill((255, 255, 255))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            pygame.draw.circle(WIN, (0, 0, 0), pygame.mouse.get_pos(), 1)
            pygame.display.update()
            pygame.event.pump()

        while pygame.mouse.get_pressed()[2]:
            pygame.draw.circle(WIN, (255, 255, 255), pygame.mouse.get_pos(), 33)
            pygame.display.update()
            pygame.event.pump()

        if pygame.key.get_pressed()[pygame.K_SPACE]:
            WIN.fill((255, 255, 255))
            pygame.display.update()

        if pygame.key.get_pressed()[pygame.K_a]:
            matrix = pygame.surfarray.array2d(WIN)
            translate_array(matrix)
            # Print all x values
            print(x_values)

            # image = np.expand_dims(correct_matrix, axis=-1).astype(np.float32)
            # scaled_image = cv2.resize(image, (28, 28))
            # ans = model.predict(scaled_image.reshape(1, 28, 28, 1))
            # print(ans.argmax())
        


if __name__ == '__main__':
    main()