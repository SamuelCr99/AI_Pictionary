import numpy as np
from PIL import Image
import tensorflow as tf
from keras import layers
import time


t = time.time()
x = []
y = []

with open('png/filelist.txt', 'r') as filelist: 
    for line in filelist:
        line = line.strip("\n")
        file_name = f'png/{line}'
        img = Image.open(file_name)
        img = img.resize((800, 800))
        x.append(np.asarray(img))
        y.append(1)
        # y.append(line.split("/")[0])

x = np.array(x)
y = np.array(y)

X_train = x[0:80]
X_test = x[80:100]
y_train = y[0:80]
y_test = y[80:100]


X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))
# X_train = X_train / 255.
# X_test = X_test / 255.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
model = tf.keras.Sequential([
    layers.Conv2D(filters=10,
                kernel_size=3, 
                activation="relu", 
                input_shape=(800,  800,  1)),
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
model.save("drawing_recognizer.h5")