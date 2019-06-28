# Training a Neural Network with Auto Keras

As you may have noticed when developing with the Keras library you still need to have a decent repository of knowledge to build a well performing neural network. AutoKeras is riding on the movement of AutoML that helps bring machine learning and computer vision to the g eneral public. 

In this demo we will be walking through a very quick example of how to train a neural network with AutoKeras using the same dataset that we gathered during the second portion of our demo. 

1. First we will need to import the required libraries.  
    ```python
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    import autokeras as ak
    from keras.preprocessing.image import load_img, img_to_array
    import os
    import pandas as pd
    import numpy as np
    ```

1. Now we will reuse some code from our previous keras script, and read the image paths and labels into a pandas dataframe for spliting and processing of our datasets. 
    ```python
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
    ```

1. Currently, AutoKeras only has a `.fit` function to train a neural network, which requries us to provide our train/validation datasets as numpy arrays forcing us to load all our data into memory. This is important to rememeber because the Keras library provides a lot of functions to help us work with larger data sets i.e. Data Generators and other fit functions.  

    Because of this we must load each image into  numpy array. Since we have a small dataset the following should suffice.   
    ```python
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
    ```

1. Now lets train a model!
    ```python
    # instantiate a classifier and fit a network
    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train, time_limit=3600)
    ```