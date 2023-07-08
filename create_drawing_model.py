import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers
import json
import random
import matplotlib.pyplot as plt

def shuffle_list(l):
    for i in range(len(l[0])):
        r = random.randint(0, len(l[0]) - 1)
        l[0][i], l[0][r] = l[0][r], l[0][i]
        l[1][i], l[1][r] = l[1][r], l[1][i]
    return l



def main():
    # Load the labels
    with open('labels.json', 'r') as f:
        labels = json.load(f)

    xy = [[], []]

    with open('png/filelist.txt', 'r') as filelist: 
        for line in filelist:
            count += 1
            line = line.strip("\n")
            file_name = f'png/{line}'
            img = Image.open(file_name)
            img = img.resize((100, 100))
            img_arr = np.asarray(img)
            if np.isnan(img_arr).any():
                # Throw error
                print('Error')
                quit()
            xy[0].append(img_arr)
            xy[1].append(labels[line.split("/")[0]])

    xy = shuffle_list(xy)
    x = np.array(xy[0])
    y = np.array(xy[1])

    
    a = int(len(x)*0.8)
    X_train = x[0:a]
    X_test = x[a:]
    y_train = y[0:a]
    y_test = y[a:]


    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    X_train = X_train / 255.
    X_test = X_test / 255.
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    model = tf.keras.Sequential([
        layers.Conv2D(filters=3,
                    kernel_size=3, 
                    activation="relu", 
                    input_shape=(100,  100,  1)),
        layers.Conv2D(3,  3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(3,  3, activation="relu"),
        layers.Conv2D(3,  3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(250, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=15, batch_size=1)
    model.save("drawing_recognizer.h5")


if __name__ == '__main__':
    main()