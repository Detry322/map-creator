from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD

import time
import numpy as np
import logging

from app.models.base import BaseModel

class Autoencoder(BaseModel):
  EPOCHS = 100
  NOISE_SIZE = 100
  MAX_BATCH_SIZE = 100
  VALIDATION_SIZE = 10

  def _construct_model(self):
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='same', subsample=(2,2), input_shape=(64,64,3)))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Convolution2D(32, 5, 5))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Convolution2D(32, 5, 5))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    model.add(Flatten())
    model.add(Dense(8*8))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8, 8, 1)))

    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(3, 5, 5, border_mode='same', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    self.model = model

  def _train_generator(self):
    while True:
      image = np.array([next(self.loader) for i in range(7)])
      yield (image, image)

  def train(self):
    validation_images = np.array([self.loader.random()/255.0 for _ in range(self.VALIDATION_SIZE)])

    checkpoint = ModelCheckpoint('autoencoder-{}.h5'.format(time.time()), monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    self.model.fit_generator(
      self._train_generator(),
      samples_per_epoch=len(self.loader),
      nb_epoch=self.EPOCHS,
      validation_data=(validation_images, validation_images),
      callbacks=[checkpoint])
