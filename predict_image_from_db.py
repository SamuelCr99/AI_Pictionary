from tensorflow import keras
import json
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

model = keras.models.load_model("drawing_recognizer.h5")

def find_key_from_value(d, value):
    for k, v in d.items():
        if v == value:
            return k

def main():
    random_index = np.random.randint(0, len(xy[0]))
    x = np.array(xy[0])

    # Show x 
    plt.imshow(x[random_index])
    plt.show()
    print(find_key_from_value(labels,model.predict(x[random_index].reshape(1, 100, 100, 1)).argmax()))


if __name__ == "__main__":
    with open('labels.json', 'r') as f:
        labels = json.load(f)

    xy = [[], []]

    c = 0
    with open('png/filelist.txt', 'r') as filelist: 
        for line in filelist:
            c += 1
            if c > 4000:
                break
            line = line.strip("\n")
            file_name = f'png/{line}'
            img = Image.open(file_name)
            img = img.resize((100, 100))
            xy[0].append(np.asarray(img))
            xy[1].append(labels[line.split("/")[0]])
    while(1):
        main()