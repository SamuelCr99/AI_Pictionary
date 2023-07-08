from tensorflow import keras
import json
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

model = keras.models.load_model("drawing_recognizer.h5")


with open('labels.json', 'r') as f:
    labels = json.load(f)

xy = [[], []]

c = 0
with open('png/filelist.txt', 'r') as filelist: 
    for line in filelist:
        c += 1
        if c > 120:
            break
        line = line.strip("\n")
        file_name = f'png/{line}'
        img = Image.open(file_name)
        img = img.resize((100, 100))
        xy[0].append(np.asarray(img))
        xy[1].append(labels[line.split("/")[0]])

x = np.array(xy[0])
# Show x 
plt.imshow(x[100])
plt.show()
print(model.predict(x[100].reshape(1, 100, 100, 1)).argmax())