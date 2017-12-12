from app.models import all_models
from app.utils import mkdir_p
from app import GENERATED_TILES_FOLDER, RANDOM_FOLDER, BACKPROPS_FOLDER

from scipy import misc

import glob
import numpy as np
import os
from keras.models import load_model, Model
from keras.optimizers import Adam, SGD, Adagrad
from keras.layers import LocallyConnected1D, Input, Reshape

from app import BACKPROPS_FOLDER, FORWARDPROPS_FOLDER, RANDOM_FOLDER
from app.utils import mkdir_p

from app.forwardprop import forwardprop_single_image

NOISE_SIZE = 100

import time 

def random(model_file):
  model = load_model(model_file)
  generator = model.layers[0]
  generator.trainable = False
  for layer in generator.layers:
    layer.trainable = False

  api_key_water = [np.loadtxt(filename) for filename in glob.glob(os.path.join(BACKPROPS_FOLDER, 'api_key', 'water', '*.txt'))]
  no_api_key_water = [np.loadtxt(filename) for filename in glob.glob(os.path.join(BACKPROPS_FOLDER, 'no_api_key', 'water', '*.txt'))]

  no_api_key_trees = np.loadtxt(os.path.join(BACKPROPS_FOLDER, 'no_api_key', 'trees', '3391.png.txt'))

  folder = os.path.join(RANDOM_FOLDER, '{}'.format(time.time()))
  mkdir_p(folder)

  for a in api_key_water:
    for na in no_api_key_water:
      api_key_trees = a - na + no_api_key_trees

      image = forwardprop_single_image(generator, api_key_trees)

      misc.imsave(os.path.join(folder, 'land-{}.png'.format(time.time())), ((image + 1)*128).astype('uint8'))
