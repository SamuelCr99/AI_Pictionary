import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
import json
from keras import layers
from sklearn.model_selection import train_test_split




x = [] 
y = []

dict = {}

c = -1
for bit_map in os.listdir('data'):
    data = np.load('data/' + bit_map, encoding='latin1', allow_pickle=True)
    c += 1
    dict[c] = bit_map[18:-4]
    for i in range(10000):
        x.append(data[i].reshape((28, 28)))
        y.append(c)



# Split x and y into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))
X_train = X_train / 255.
X_test = X_test / 255.

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('uint8')
y_test = y_test.astype('uint8')

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(10,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(20,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(20,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(20,activation='relu'))
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