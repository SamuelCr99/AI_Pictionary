import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import json
from keras import layers
from sklearn.model_selection import train_test_split


def create_model(train_data_location):
    """
    Creates a convolutional neural network for predicting doodles

    Parameters: 
    train_data_location[str]: The file path of the training data. This training 
    data should be numpy bitmaps. Bitmaps can be downloaded here: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
    
    
    """

    # Lists for holding data and labels
    x = []
    y = []

    # Dictionary mapping between labels and numbers 
    dict = {}

    c = -1
    for bit_map in os.listdir(train_data_location):
        data = np.load(f"{train_data_location}/{bit_map}", encoding='latin1', allow_pickle=True)
        c += 1
        dict[c] = bit_map[18:-4]
        # Trains each label on 10000 images
        for i in range(10000):
            x.append(data[i].reshape((28, 28)))
            y.append(c)


    # Split x and y into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    # Cast all lists to np arrays 
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    # Normalize the values 
    X_train = X_train / 255.
    X_test = X_test / 255.

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('uint8')
    y_test = y_test.astype('uint8')

    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(20, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(20, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(c+1, activation='softmax'))


    model.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=4, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(test_acc)

    model.save("drawing_recognizer.h5")

    # Dump the dict to a json file
    with open('num_to_image_dict.json', 'w') as f:
        json.dump(dict, f, indent=4)

if __name__ == "__main__":
    create_model("data")