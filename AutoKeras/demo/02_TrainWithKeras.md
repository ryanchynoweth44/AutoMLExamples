# Training a Neural Network with Keras

Keras is a popular library that assists data scientists by providing a high-level interface to run neural networks based on top of TensorFlow, CNTK, and Theano. We will walk through a very basic example of training and configuring a convolution neural network to classify images. 

By providing this portion of the demo I hope to show how AutoKeras takes the next step in making deep learning available to the masses.  

1. First lets import all the libraries we will need. 
    ```python
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
    from keras import backend as K
    from keras import callbacks as cb
    from PIL import Image
    import os
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    ```


1. Next we will need to instantiate variables we need to train a neural . 
    ```
    output_dir = 'keras_model/'
    os.makedirs(output_dir, exist_ok=True)
    bs = 16
    epochs = 25
    img_width, img_height = 256, 256
    nb_channels = 3

    if K.image_data_format() == 'channels_first':
        input_shape = (nb_channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, nb_channels)
    ```

1. Next Keras has an ImageDataGenerator class that allows us to convert our images into datasets that can be ingested by our algorithm.  
    ```python
    # load data
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        rescale=1./255,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    ```

1. There are many different ways to organize your training dataset in order to use built in keras functions to train a CNN. In our case we have a single folder for all images in a class. Since we will be using the `flow_from_dataframe` function to generate our datasets we need to organize the file names, file paths, and labels. We will split our image dataframe into a train, validate, and test dataset. 
    ```python
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

    train, validate, test = np.split(images.sample(frac=1), [int(.6*len(images)), int(.8*len(images))])
    train.head()
    len(train)
    len(validate)
    len(test)
    ```

1. We will now use our datasets to generate our datasets that can be consumed by our neural network.  
    ```python

    train_generator = datagen.flow_from_dataframe(
        dataframe=train,
        directory="./",
        x_col="filepath",
        y_col="label",
        class_mode="categorical",
        batch_size=bs
    )

    validate_generator = datagen.flow_from_dataframe(
        dataframe=validate,
        directory="./",
        x_col="filepath",
        y_col="label",
        class_mode="categorical",
        batch_size=bs
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=test,
        directory="./",
        x_col="filepath",
        y_col="label",
        class_mode="categorical",
        batch_size=bs
    )
    ```

1. Now we will build the architecture of our network. 
    ```python
    # Deep model with limited features/convolutions
    model_prefix = 'image_model_'

    model = Sequential()
    model.add(Conv2D(8, (2, 2), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])

    ```

1. We will want to checkpoint our progress to save our model weights as we train, commit to early stopping to avoid overfitting to our training dataset, and establish logging for our training. 
    ```python
    # create some callbacks to generate historical metrics
    csv_logger = cb.CSVLogger(output_dir + model_prefix + 'training.log')
    early_stop = cb.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
    checkpoints = cb.ModelCheckpoint(filepath= output_dir + model_prefix + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, 
                                    save_best_only=True, save_weights_only=False, 
                                    mode='auto', period=1)
    loggers = [csv_logger, early_stop,checkpoints]

    ```

1. Now its time to train a CNN!
    ```python
    ############### train our cnn ################
    nb_train_samples = 5000
    nb_validation_samples = 1000


    training = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // bs,
        epochs=epochs,
        validation_data=validate_generator,
        validation_steps=nb_validation_samples // bs,
        callbacks=loggers)
    ```

1. Quickly save our model and weights to file. 
    ```python

    os.makedirs(output_dir, exist_ok=True)
    # save the models and weights to file
    weight_out = os.path.join(output_dir, 'image_epochs.h5')
    model_out = os.path.join(output_dir, 'image_model.h5')
    model.save_weights(weight_out)
    model.save(model_out)
    ```
1. Predict on our test dataset. 
    ```python 
    ### Test dataset predictions
    probabilities = model.predict_generator(test_generator, steps=len(test_generator.filenames)/bs, verbose=1)

    predictions = pd.DataFrame({
        "filename": test_generator.filenames,
        "label": test_generator.classes,
        "prediction": np.argmax(probabilities, axis=1)
        })

    predictions.head()
    ```

1. See overall accuracy (ignore the poor performance), and see confusion matrix.  
    ```python
    ## confusion matrix
    df_confusion = pd.crosstab(predictions['label'], predictions['prediction'])
    df_confusion

    acc = (len(predictions[predictions['label'] == predictions['prediction']]))/len(predictions)
    acc
    ```