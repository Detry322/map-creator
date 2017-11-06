from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD

import numpy as np
import logging
import time

from app.models.base import BaseModel

class BasicDCGAN(BaseModel):
  EPOCHS = 1000
  NOISE_SIZE = 100
  MAX_BATCH_SIZE = 512
  GENERATOR_OPTIMIZER = 'adam'
  DISCRIMINATOR_OPTIMIZER = SGD(lr=0.0005, momentum=0.9, nesterov=True)
  FULL_OPTIMIZER = SGD(lr=0.0005, momentum=0.9, nesterov=True)

  def _construct_model(self):
    self.generator_model = self._construct_generator()
    self.discriminator_model = self._construct_discriminator()
    self.model = self._construct_full(self.generator_model, self.discriminator_model)
    self._compile()

  def _construct_from_file(self, filename):
    self.model = load_model(filename)
    self.generator_model = self.model.layers[0]
    self.discriminator_model = self.model.layers[1]
    self._compile()

  def _construct_generator(self):
    model = Sequential()
    model.add(Dense(input_dim=self.NOISE_SIZE, output_dim=(16*16), activation='relu'))
    model.add(Reshape((16, 16, 1)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, 5, 5, border_mode='same', activation='sigmoid'))
    return model

  def _construct_discriminator(self):
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2,2), input_shape=self.image_size))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(64, 5, 5, border_mode='same', subsample=(2,2)))
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(1, activation='sigmoid'))
    return model

  def _construct_full(self, generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

  def _compile(self):
    self.generator_model.compile(loss='binary_crossentropy', optimizer=self.GENERATOR_OPTIMIZER, metrics=['accuracy'])
    self.discriminator_model.compile(loss='binary_crossentropy', optimizer=self.DISCRIMINATOR_OPTIMIZER, metrics=['accuracy'])
    self.model.compile(loss='binary_crossentropy', optimizer=self.FULL_OPTIMIZER, metrics=['accuracy'])

  def _generate_batch(self, num):
    noise = np.random.uniform(-1, 1, (num, self.NOISE_SIZE))
    return self.generator_model.predict(noise)

  def generate_image(self):
    return (self._generate_batch(1)[0]*256.0).astype('uint8')

  def train(self):
    model_name = "basic_dcgan-{}.h5".format(time.time())
    for epoch in range(self.EPOCHS):
      logging.info("=== Epoch {}".format(epoch))
      for batch_base in range(0, len(self.image_loader), self.MAX_BATCH_SIZE):
        batch_size = min(len(self.image_loader) - batch_base, self.MAX_BATCH_SIZE)
        logging.info("Training {} images".format(batch_size))

        # first, train discriminator
        self.discriminator_model.trainable = True
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=self.DISCRIMINATOR_OPTIMIZER, metrics=['accuracy'])
        images = np.array([next(self.image_loader)/255.0 for _ in range(batch_size)])
        generated_images = self._generate_batch(batch_size)
        discriminator_X = np.concatenate((images, generated_images))
        discriminator_Y = np.array([1]*batch_size + [0]*batch_size)
        discriminator_loss = self.discriminator_model.train_on_batch(discriminator_X, discriminator_Y)
        logging.info("Discriminator Loss: {}".format(discriminator_loss))

        # next, train generator
        self.discriminator_model.trainable = False
        self.model.compile(loss='binary_crossentropy', optimizer=self.FULL_OPTIMIZER, metrics=['accuracy'])
        full_X = np.random.uniform(-1, 1, (batch_size, self.NOISE_SIZE))
        full_Y = np.array([1]*batch_size)
        full_loss = self.model.train_on_batch(full_X, full_Y)
        logging.info("Full loss: {}".format(full_loss))
      self.model.save(model_name)
