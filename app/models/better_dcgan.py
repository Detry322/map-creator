from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.backend import clear_session

import numpy as np
import os
import logging
import time
import tempfile

from app.models.base import BaseModel

class BetterDCGAN(BaseModel):
  EPOCHS = 1000
  NOISE_SIZE = 100
  MAX_BATCH_SIZE = 256

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
    model.add(Dense(input_dim=self.NOISE_SIZE, units=(4*4*1024), kernel_initializer='random_normal'))
    model.add(Reshape((4, 4, 1024)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2DTranspose(512, 5, strides=2, padding='same', kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2DTranspose(256, 5, strides=2, padding='same', kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2DTranspose(128, 5, strides=2, padding='same', kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2DTranspose(3, 5, strides=2, padding='same', kernel_initializer='random_normal', activation='sigmoid'))
    return model

  def _construct_discriminator(self):
    model = Sequential()
    model.add(Conv2D(64, 5, strides=2, padding='same', kernel_initializer='random_normal', input_shape=self.image_size))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, 5, strides=2, padding='same', kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, 5, strides=2, padding='same', kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, 5, strides=2, padding='same', kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4*4*512,)))
    model.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))
    return model

  def _construct_full(self, generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

  def _compile(self):
    self.generator_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002, beta_1=0.5), metrics=['accuracy'])
    self.discriminator_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002, beta_1=0.5), metrics=['accuracy'])
    self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002, beta_1=0.5), metrics=['accuracy'])

  def _reset_memory(self):
    logging.info("=== Resetting memory footprint")
    t, name = tempfile.mkstemp()
    os.close(t)
    self.model.save(name)
    clear_session()
    self._construct_from_file(name)
    os.unlink(name)

  def _generate_batch(self, num):
    noise = np.random.uniform(-1, 1, (num, self.NOISE_SIZE))
    return self.generator_model.predict(noise)

  def generate_image(self):
    return (self._generate_batch(1)[0]*256.0).astype('uint8')

  def train(self):
    model_name = "better_dcgan-{}.h5".format(time.time())
    i = 0
    for epoch in range(self.EPOCHS):
      logging.info("=== Epoch {}".format(epoch))
      for batch_base in range(0, len(self.image_loader), self.MAX_BATCH_SIZE):
        i += 1
        if i % 6 == 0:
          self._reset_memory()

        batch_size = min(len(self.image_loader) - batch_base, self.MAX_BATCH_SIZE)
        logging.info("Training {} images".format(batch_size))

        # first, train discriminator
        self.discriminator_model.trainable = True
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002, beta_1=0.5), metrics=['accuracy'])
        images = np.array([next(self.image_loader)/255.0 for _ in range(batch_size)])
        generated_images = self._generate_batch(batch_size)
        discriminator_X = np.concatenate((images, generated_images))
        discriminator_Y = np.array([1]*batch_size + [0]*batch_size)
        discriminator_loss = self.discriminator_model.train_on_batch(discriminator_X, discriminator_Y)
        logging.info("Discriminator Loss: {}".format(discriminator_loss))

        # next, train generator
        self.discriminator_model.trainable = False
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.002, beta_1=0.5), metrics=['accuracy'])
        full_X = np.random.uniform(-1, 1, (batch_size, self.NOISE_SIZE))
        full_Y = np.array([1]*batch_size)
        full_loss = self.model.train_on_batch(full_X, full_Y)
        logging.info("Full loss: {}".format(full_loss))
      logging.info("=== Writing model to disk")
      self.model.save(model_name)
