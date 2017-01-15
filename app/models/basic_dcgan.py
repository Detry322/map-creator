from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD

import numpy as np
import logging

class BasicDCGAN:
  EPOCHS = 100
  NOISE_SIZE = 100
  MAX_BATCH_SIZE = 2
  GENERATOR_OPTIMIZER = 'adam'
  DISCRIMINATOR_OPTIMIZER = SGD(lr=0.0005, momentum=0.9, nesterov=True)
  FULL_OPTIMIZER = SGD(lr=0.0005, momentum=0.9, nesterov=True)

  def __init__(self, loader, from_file=None):
    self.loader = loader
    if from_file is not None:
      raise NotImplementedError("RIP")
    self._construct_model()

  def _construct_model(self):
    self.generator_model = self._construct_generator()
    self.discriminator_model = self._construct_discriminator()
    self.full_model = self._construct_full(self.generator_model, self.discriminator_model)

  def _construct_generator(self):
    model = Sequential()
    model.add(Dense(input_dim=self.NOISE_SIZE, output_dim=(64*64), activation='relu'))
    model.add(Reshape((64, 64, 1)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=self.GENERATOR_OPTIMIZER, metrics=['accuracy'])
    return model

  def _construct_discriminator(self):
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2,2), input_shape=(256,256,3)))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=self.DISCRIMINATOR_OPTIMIZER, metrics=['accuracy'])
    return model

  def _construct_full(self, generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=self.FULL_OPTIMIZER, metrics=['accuracy'])
    return model

  def generate_batch(self, num):
    noise = np.random.uniform(-1, 1, (num, self.NOISE_SIZE))
    return self.generator_model.predict(noise)

  def generate_image(self):
    raise NotImplementedError("RIP")

  def train(self):
    for epoch in range(self.EPOCHS):
      logging.info("=== Epoch {}".format(epoch))
      for batch_base in range(0, len(self.loader), self.MAX_BATCH_SIZE):
        batch_size = min(len(self.loader) - batch_base, self.MAX_BATCH_SIZE)
        logging.info("Training {} images".format(batch_size))

        # first, train discriminator
        self.discriminator_model.trainable = True
        images = np.array([next(self.loader)/255.0 for _ in range(batch_size)])
        generated_images = self.generate_batch(batch_size)
        discriminator_X = np.concatenate((images, generated_images))
        discriminator_Y = np.array([0]*batch_size + [1]*batch_size)
        discriminator_loss = self.discriminator_model.train_on_batch(discriminator_X, discriminator_Y)
        logging.info("Discriminator Loss: {}".format(discriminator_loss))

        # next, train generator
        self.discriminator_model.trainable = False
        full_X = np.random.uniform(-1, 1, (batch_size, self.NOISE_SIZE))
        full_Y = np.array([0]*batch_size)
        full_loss = self.full_model.train_on_batch(full_X, full_Y)
        logging.info("Full loss: {}".format(full_loss))
