import pygame
from tensorflow import keras
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


WIN = pygame.display.set_mode((800,800))

def translate_colors(colors):
    for i in range(len(colors)):
        for j in range(len(colors[i])):
            if colors[i][j] == 16777215:
                colors[i][j] = 0
            else:
                colors[i][j] = 1
    return colors


def main():
    model = keras.models.load_model("drawing_recognizer.h5")
    # Read the json file
    with open('num_to_image_dict.json', 'r') as f:
        data = json.load(f)

    WIN.fill((255, 255, 255))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            # Draw a rect around the mouse
            r = pygame.Rect(pygame.mouse.get_pos(), (10, 10))
            pygame.draw.circle(WIN, (0, 0, 0), pygame.mouse.get_pos(), 10)
            pygame.display.update()
            pygame.event.pump()

        while pygame.mouse.get_pressed()[2]:
            pygame.draw.circle(WIN, (255, 255, 255), pygame.mouse.get_pos(), 12)
            pygame.display.update()
            pygame.event.pump()

        if pygame.key.get_pressed()[pygame.K_SPACE]:
            WIN.fill((255, 255, 255))
            pygame.display.update()

        if pygame.key.get_pressed()[pygame.K_a]:
            matrix = pygame.surfarray.array2d(WIN)
            correct_matrix = translate_colors(matrix).transpose()
            image = np.expand_dims(correct_matrix, axis=-1).astype(np.float32)
            image = cv2.resize(image, (28, 28))
            # plt.imshow(image)
            # plt.show()

            ans = model.predict(image.reshape(1, 28, 28, 1))
            print(ans)
            print(data[str(ans.argmax())])
        


if __name__ == '__main__':
    main()