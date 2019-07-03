# Training a Neural Network with Auto Keras

As you may have noticed when developing with the Keras library you still need to have a decent repository of knowledge to build a well performing neural network. AutoKeras is riding on the movement of AutoML that helps bring machine learning and computer vision to the g eneral public. 

In this demo we will be walking through a very quick example of how to train a neural network with AutoKeras using the same dataset that we gathered during the second portion of our demo. 

1. First we will need to import the required libraries.  
    ```python
    from sklearn.model_selection import train_test_split
    import autokeras as ak
    import os
    import pandas as pd
    import numpy as np
    from PIL import Image, ImageOps
    ```

1. Now we will reuse some code from our previous keras script, add a few lines to remove some of the most difficult file extensions, and read the image paths and labels into a pandas dataframe for spliting and processing of our datasets. 
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

    # extract file extensions
    images["file_ext"] = images.apply(lambda x: os.path.splitext(x['filename'])[1], axis=1 )
    images.head()

    # need to get rid of bad file extensions
    images["file_ext"].unique()
    images = images.loc[ images['file_ext'].isin(['.jpg', '.png', '.jpeg', '.JPG', '.img']) ]
    images["file_ext"].unique()


    train, test = train_test_split(images, test_size=0.2 )
    ```

1. Currently, AutoKeras only has a `.fit` function to train a neural network, which requries us to provide our train/validation datasets as numpy arrays forcing us to load all our data into memory. This is important to rememeber because the Keras library provides a lot of functions to help us work with larger data sets i.e. Data Generators and other fit functions.  

    Because of this we must load each image into  numpy array. Since we have a small dataset the following should suffice.   
    ```python
    # get the file paths we want to load
    files = np.array(train.filepath)
    labels = np.array(train.label)
    x_train = np.empty((len(files), 200, 200, 3), int)
    y_train = []

    # load each image and convert it to an ndarray
    # append each array to a list
    for i in range(0, len(files)):    
        print(str(i) + "/" + str(len(files)-1))
        x = Image.open(files[i])
        x2 = ImageOps.fit(x, (200,200), Image.ANTIALIAS)
        arr = np.array(x2)
        
        x_train[i] = arr
        
        y_train.append(labels[i])

    y_train = np.array(y_train)
    ```

1. Now lets train a model! I am adding in a few timestamps to see how long it takes to train a model. Note that the final fit model gets the best model we trained and fits it to the `clf` object for us to save out to a file.   
    ```python
    # instantiate a classifier and fit a network
    clf = ak.ImageClassifier(verbose=True)

    import datetime as dt
    t = dt.datetime.now()
    print("Starting time: " + str(t))
    clf.fit(x_train, y_train, time_limit=3600)
    print("End time: " + str(dt.datetime.now()))
    print("Total time: " + str(dt.datetime.now() - t))
    ```

1. We will want to evaluate our model on our test dataset as well. So we will need to format our test data as a numpy array and apply our model to the dataset.  
    ```python
    # get the file paths we want to load
    files = np.array(test.filepath)
    labels = np.array(test.label)
    x_test = np.empty((len(files), 200, 200, 3), int)
    y_test = []

    # load each image and convert it to an ndarray
    # append each array to a list
    for i in range(0, len(files)):    
        print(str(i) + "/" + str(len(files)-1))
        x = Image.open(files[i])
        x2 = ImageOps.fit(x, (200,200), Image.ANTIALIAS)
        arr = np.array(x2)
        
        x_test[i] = arr
        
        y_test.append(labels[i])

    y_test = np.array(y_test)

    clf.final_fit(X_train, Y_train, X_test, Y_test, retrain=True)

    # apply model
    y_pred = clf.evaluate(x_test, y_test)
    print(y_pred)
    ```

1. Finally, we will want to export and save our model to file. 
    ```python
    os.makedirs('autokeras_model', exist_ok=True)

    clf.export_autokeras_model('autokeras_models/autokeras_images_retrained.h5')
    clf.export_keras_model('models/keras_images_retrained.h5')
    clf.load_searcher().load_best_model().produce_keras_model().save('models/keras_model_1_image_retrained.h5')
    ```

