from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import autokeras as ak
from keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
import numpy as np


# pandas dataframes for datasets with image paths
## read file names
image_dirs = [f for f in os.listdir("./data") if '.csv' not in f]

images = pd.DataFrame(columns=["filename", "filepath", "label"])

for dir in image_dirs:
   images = images.append(pd.DataFrame({
    "filename": os.listdir("./data/"+dir),
    "filepath": ["data/"+dir+"/"+ s for s in os.listdir("./data/"+dir) ],
    "label": dir
    }))

images.head()


train, test = train_test_split(images, test_size=0.2 )


# get the file paths we want to load
files = np.array(train.filepath)
x_train = []

# load each image and convert it to an ndarray
# append each array to a list
for i in range(0, len(files)):    
    print(str(i) + "/" + str(len(files)))
    x = img_to_array(load_img(files[1]))
    x_train.append(x)

# convert the list of arrays to an array
x_train = np.array(x_train)
# convert our labels to numpy array
y_train = np.array(train.label)

# instantiate a classifier and fit a network
clf = ak.ImageClassifier()

import datetime as dt
t = dt.datetime.now()
print("Starting time: " + str(t))
clf.fit(x_train, y_train, time_limit=10800)
print("End time: " + str(dt.datetime.now()))
print("Total time: " + str(dt.datetime.now() - t))

