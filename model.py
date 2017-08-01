import csv
import os
import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

data_dirs = ['sample_data', 'data_bridge_and_dirtroad', 'recovery_drive', 'data_lap']
# data_dirs = ['data_lap', 'data_bridge_and_dirtroad']

use_side_images = False
do_augmentation = True
use_model = 'nvidia'

images = []
measurements = []

for data_dir in data_dirs:
    lines1 = []
    with open(os.path.join(data_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines1.append(line)

    for line in tqdm(lines1[1:]):
        source_path = line[0]
        filename = os.path.basename(source_path)
        current_path = os.path.join(data_dir, 'IMG', filename)
        image = cv2.imread(current_path)
        image = image[:, :, ::-1]
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

    if use_side_images:
        for line in tqdm(lines1[1:]):
            for i in range(2):
                source_path = line[i + 1]
                filename = os.path.basename(source_path)
                current_path = os.path.join(data_dir, 'IMG', filename)
                image = cv2.imread(current_path)
                image = image[:, :, ::-1]
                images.append(image)
                if i == 0:
                    measurement = float(line[3]) + 0.2
                else:
                    measurement = float(line[3]) - 0.2
                measurements.append(measurement)

print('*****************{} images'.format(len(images)))

if do_augmentation:
    augmented_images, augmented_measurements = [], []
    for image, measurement in tqdm(zip(images, measurements)):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement * -1.0)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
else:
    X_train = np.array(images)
    y_train = np.array(measurements)

print('*****************{} images'.format(X_train.shape[0]))

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

if use_model == 'lenet':  # lenet
    model_name = 'lenet'
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    adam_opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='mse', optimizer=adam_opt)
elif use_model == 'nvidia':
    model_name = 'nvidia'
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    adam_opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    model.compile(loss='mse', optimizer=adam_opt)
else:
    model_name = 'inception'
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    N_layers = len(model.layers)
    print('model has {} layers'.format(N_layers))

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False

    # N_last = min(N_layers, n_traiable_layers)
    # print('setting last {} layers to be trainable'.format(N_last))
    # for layer in model.layers:
    #     layer.trainable = False
    # for layer in model.layers[-N_last:]:
    #     layer.trainable = True
    for layer in base_model.layers:
        layer.trainable = False

    adam_opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
    model.compile(loss='mse', optimizer=adam_opt)


model_fname = 'model_{}.h5'.format(model_name)
print('model to be saved as {}'.format(model_fname))
checkpoint = ModelCheckpoint(model_fname, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='auto')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=16)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=256, epochs=100,
          callbacks=[checkpoint, earlystop])
