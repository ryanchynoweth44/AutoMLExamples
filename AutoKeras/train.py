import pip

pip.main(['install', "Pillow"])
pip.main(['install', "azure"])
pip.main(['install', "keras"])
pip.main(['install', "h5py"])
pip.main(['install', "tensorflow-gpu"])
# pip.main(['install', "https://cntk.ai/PythonWheel/GPU-1bit-SGD/cntk-2.1-cp35-cp35m-win_amd64.whl"])
pip.main(['install', "https://cntk.ai/PythonWheel/GPU/cntk-2.1-cp35-cp35m-linux_x86_64.whl"])

# print("Pip installs are done.")

import sys
import os
# Import necessary items
os.environ["KERAS_BACKEND"] = "cntk"
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import metrics
from keras import callbacks as cb
import itertools
import numpy as np
# import azure file storage libraries
import azure
import azure.storage
import azure.storage.file
from azure.storage.file import FileService


# Set variables
# target dimensions of our images - originals are 800 x 800 pixels
img_width, img_height = 200, 200 # Dropping 3/4 of the pixels seems like very little impact to images (they are 'blocky')

# Set the training/test/validate directories
data_path = sys.argv[1]
output_dir = sys.argv[2]


train_data_dir = data_path + "/train"
test_data_dir = data_path + "/test"
validation_data_dir = data_path + "/validate"

# setting hyper paramters 
epochs = 100
batch_size = 16
greyscale = True

#assume RGB
nb_channels = 3  
if (greyscale): nb_channels = 1 

if K.image_data_format() == 'channels_first':
    input_shape = (nb_channels, img_width, img_height)
else:
    input_shape = (img_width, img_height, nb_channels)


############ Build the cnn ############
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


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_features = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')


validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')

test_features = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale')



# based on the results above, populate this
nb_train_samples = 5000
nb_validation_samples = 1000

# create some callbacks to generate historical metrics
csv_logger = cb.CSVLogger(output_dir + model_prefix + 'training.log')
early_stop = cb.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
checkpoints = cb.ModelCheckpoint(filepath= output_dir + model_prefix + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, 
                                 save_best_only=True, save_weights_only=False, 
                                 mode='auto', period=1)
loggers = [csv_logger, early_stop,checkpoints]


############### train our cnn ################
training = model.fit_generator(
    train_features,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=loggers)

# save the models and weights to file
weight_out = os.path.join(output_dir, 'image_epochs.h5')
model_out = os.path.join(output_dir, 'image_model.h5')
model.save_weights(weight_out)
model.save(model_out)

aname = "gfdemodata"
akey = "dyw2/M9WIIxfz5Y4tFEFpkcnui7niUeIPADqUGMeRX8IDWEyVk+Rnme5eF6Fh2HAcB3g8F7OUzf+dz9fRMdW4w=="
share = "data"

## save model and weights to latest folder
fls = FileService(account_name=aname, account_key=akey)
fls.create_file_from_path(
    share,
    None,
    'azurefileshare/outputs/01_latest/image_epochs.h5',
    weight_out
    )
fls.create_file_from_path(
    share,
    None,
    'azurefileshare/outputs/01_latest/image_model.h5',
    model_out
    )


## Test our model on the test images
nb_batches_to_capture = 500 # batches * batch_size should be less than total number of items available, else repeats will happen
y_true = []
y_pred = []

## batch predictions
for index in range(nb_batches_to_capture):
    next_batch = next(test_features)
    images = next_batch[0]
    categories = next_batch[1]

    # To get y_pred, we actually need to predict the categories of the all_images set
    predicted_categories= model.predict_classes(x=images,batch_size=batch_size)
    predicted_probs = model.predict_proba(verbose=True, x=images,batch_size=batch_size)
   
    if len(categories) == len(predicted_categories):
        y_true.extend(categories)
        y_pred.extend(predicted_categories)
    else:
        print("Mismatched actual and predicted - ignoring batch")



import pandas as pd
# Convert to dataframes and output a scored dataset
labels = pd.DataFrame(predicted_categories, columns = ['PredictedLabels'])
probs = pd.DataFrame(predicted_probs, columns = ['PredictedHSC', 'PredictedHSO', 'PredictedNHS'])
names = pd.DataFrame(test_features.filenames, columns = ['FileName'])
output = pd.merge(left = names, left_index = True, right = labels, right_index = True, how = 'inner')
output = pd.merge(left = output, left_index = True, right = probs, right_index = True, how = 'inner')
output['PredictedLabelCat'] = 'HSC'
output['PredictedLabelCat'][output['PredictedLabels'] == 1] = 'HSO'
output['PredictedLabelCat'][output['PredictedLabels'] == 2] = 'NHS'

import datetime 
datestring = datetime.date.today().strftime("%Y%m%d")

output_str = datestring + "_testdataset.csv"
output.to_csv(os.path.join(output_dir, output_str))
