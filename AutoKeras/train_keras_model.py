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

output_dir = 'keras_model/'
bs = 16
epochs = 25
img_width, img_height = 256, 256
greyscale = False
nb_channels = 3
if greyscale: np_channels = 1

if K.image_data_format() == 'channels_first':
    input_shape = (nb_channels, img_width, img_height)
else:
    input_shape = (img_width, img_height, nb_channels)

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
    train_generator,
    steps_per_epoch=nb_train_samples // bs,
    epochs=epochs,
    validation_data=validate_generator,
    validation_steps=nb_validation_samples // bs,
    callbacks=loggers)

os.makedirs(output_dir, exist_ok=True)
# save the models and weights to file
weight_out = os.path.join(output_dir, 'image_epochs.h5')
model_out = os.path.join(output_dir, 'image_model.h5')
model.save_weights(weight_out)
model.save(model_out)


### Test dataset predictions
probabilities = model.predict_generator(test_generator, steps=len(test_generator.filenames)/bs, verbose=1)

predictions = pd.DataFrame({
    "filename": test_generator.filenames,
    "label": test_generator.classes,
    "prediction": np.argmax(probabilities, axis=1)
    })

predictions.head()



## confusion matrix
df_confusion = pd.crosstab(predictions['label'], predictions['prediction'])

acc = (len(predictions[predictions['label'] == predictions['prediction']]))/len(predictions)
acc

