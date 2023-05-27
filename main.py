import pygame
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import layers
import matplotlib.pyplot as plt

WIN = pygame.display.set_mode((28, 28))


def create_model():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    X_train = X_train / 255.
    X_test = X_test / 255.
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    model = tf.keras.Sequential([
        layers.Conv2D(filters=10,
                    kernel_size=3, 
                    activation="relu", 
                    input_shape=(28,  28,  1)),
        layers.Conv2D(10,  3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(10,  3, activation="relu"),
        layers.Conv2D(10,  3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10)
    

def main():
    create_model()
    WIN.fill((255, 255, 255))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        while pygame.mouse.get_pressed()[0]:
            pygame.draw.circle(WIN, (0, 0, 0), pygame.mouse.get_pos(), 5)
            pygame.display.update()
            pygame.event.pump()

        if pygame.key.get_pressed()[pygame.K_SPACE]:
            WIN.fill((255, 255, 255))
            pygame.display.update()

        if pygame.key.get_pressed()[pygame.K_a]:
            pass
        


if __name__ == '__main__':
    main()