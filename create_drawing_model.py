import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers
import json
import random
import matplotlib.pyplot as plt

def shuffle_list(l):
    new_xy = [[], []]

    while(len(l[0]) > 0):
        r = random.randint(0, len(l[0]) - 1)
        new_xy[0].append(l[0][r])
        new_xy[1].append(l[1][r])
        l[0].pop(r)
        l[1].pop(r)
    
    return new_xy


def convert_image_to_bw(img):
    img = img.resize((100, 100))

    arr = np.array(img)

    for i in range(len(arr)):
        for k in range(len(arr[i])):
            if arr[i][k] == 255:
                arr[i][k] = 1.0
            else:
                arr[i][k] = 0.0

    return arr


def main():
    # Load the labels
    with open('labels.json', 'r') as f:
        labels = json.load(f)

    xy = [[], []]
    count = 0
    unique_labels = []
    with open('png/filelist.txt', 'r') as filelist: 
        for line in filelist:
            count += 1
            # if len(unique_labels) == 4000:
            #     break
            # if count >= 1600:
            #     count = 0
            #     continue
            # if count > 320:
            #     continue
            if count >= 4000:
                break
            line = line.strip("\n")
            file_name = f'png/{line}'
            img = Image.open(file_name)
            img_arr = convert_image_to_bw(img)
            xy[0].append(img_arr)
            xy[1].append(labels[line.split("/")[0]])

            xy[0].append(np.flip(img_arr))
            xy[1].append(labels[line.split("/")[0]])
            unique_labels.append(line.split("/")[0])

    with open('unique_label.txt', 'w') as f:
        unique_labels = list(set(unique_labels))
        for item in unique_labels:
            f.write(f'{item}\n')

    x = np.array(xy[0])
    y = np.array(xy[1])

    a = int(len(x)*0.8)
    X_train = x[0:a]
    X_test = x[a:]
    y_train = y[0:a]
    y_test = y[a:]


    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(52, activation='softmax')
    ])

    model.compile(loss="sparse_categorical_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=8, batch_size=32)

    model.save("drawing_recognizer.h5")


if __name__ == '__main__':
    main()