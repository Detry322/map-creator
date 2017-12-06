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
from scipy import misc

from app.models.base import BaseModel
from app.utils import mkdir_p
from app import GENERATED_TILES_FOLDER

class BestDCGAN(BaseModel):
  EPOCHS = 1000
  NOISE_SIZE = 100
  MAX_BATCH_SIZE = 256

  def _construct_model(self):
    self.trainable_discriminator = self._construct_discriminator()
    self.untrainable_discriminator = self._construct_discriminator()
    self.generator = self._construct_generator()
    self.model = self._construct_full(self.generator, self.untrainable_discriminator)
    self._compile()

  def _construct_generator(self):
    model = Sequential()
    model.add(Dense(input_dim=self.NOISE_SIZE, units=(4*4*1024)))
    model.add(Reshape((4, 4, 1024)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(512, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(256, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(128, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(3, 5, strides=2, padding='same', activation='tanh'))
    return model

  def _construct_discriminator(self):
    model = Sequential()
    model.add(Conv2D(64, 5, strides=2, padding='same', input_shape=self.image_size))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4*4*512,)))
    model.add(Dense(1, activation='sigmoid'))
    return model

  def _construct_full(self, generator, discriminator):
    discriminator.trainable = False
    for layer in discriminator.layers:
        layer.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

  def _compile(self):
    self.trainable_discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.5), metrics=['accuracy'])
    self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])

  def _copy_weights(self):
    self.untrainable_discriminator.set_weights(self.trainable_discriminator.get_weights())

  def _generate_noise(self, num):
    return np.random.normal(0.0, 1.0, (num, self.NOISE_SIZE))

  def _generate_batch(self, num):
    return self.generator.predict(self._generate_noise(num))

  def generate_image(self):
    return ((self._generate_batch(1)[0] + 1)*128.0).astype('uint8')

  def _load_batch(self, size):
    return np.array([(next(self.image_loader)/127.5) - 1 for _ in range(size)])

  def train(self):
    i = 0
    model_name = "best_dcgan-{}.h5".format(time.time())
    folder = os.path.join(GENERATED_TILES_FOLDER, model_name)
    mkdir_p(folder)

    for epoch in range(self.EPOCHS):
      logging.info("=== Epoch {}".format(epoch))
      for batch_base in range(0, len(self.image_loader), self.MAX_BATCH_SIZE):
        i += 1

        batch_size = min(len(self.image_loader) - batch_base, self.MAX_BATCH_SIZE)
        logging.info("Training {} images".format(batch_size))

        generated_images_batch_size = batch_size
        generated_images_X = self._generate_batch(generated_images_batch_size)
        generated_images_Y = np.array([0.0]*generated_images_batch_size)
        gen_loss = self.trainable_discriminator.train_on_batch(generated_images_X, generated_images_Y)
        logging.info("Discriminator gen. loss: {}".format(gen_loss))

        # first, train discriminator
        real_images_batch_size = batch_size
        real_images_X = self._load_batch(real_images_batch_size)
        real_images_Y = np.array([1.0]*real_images_batch_size)
        real_loss = self.trainable_discriminator.train_on_batch(real_images_X, real_images_Y)
        logging.info("Discriminator real loss: {}".format(real_loss))

        logging.info("Copying weights...")
        self._copy_weights()

        generator_batch_size = batch_size
        generator_X = self._generate_noise(generator_batch_size)
        generator_Y = np.array([1.0]*generator_batch_size)
        generator_loss = self.model.train_on_batch(generator_X, generator_Y)
        logging.info("Generator loss: {}".format(generator_loss))

        logging.info("Generating image...")
        filename = os.path.join(folder, '{}.png'.format(i))
        misc.imsave(filename, self.generate_image())

      logging.info("=== Writing model to disk")
      self.model.save(model_name)
