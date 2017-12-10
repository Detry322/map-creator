from app.models import all_models
from app.utils import mkdir_p
from app import GENERATED_TILES_FOLDER

from scipy import misc

import glob
import numpy as np
import os
from keras.models import load_model, Model
from keras.optimizers import Adam, SGD, Adagrad
from keras.layers import LocallyConnected1D, Input, Reshape

from app import BACKPROPS_FOLDER, FORWARDPROPS_FOLDER
from app.utils import mkdir_p

NOISE_SIZE = 100

def forwardprop_single_image(generator, inp):
  return generator.predict(np.array([inp]))[0]


def forwardprop(model_file, zoom):
  input_files = glob.glob(os.path.join(BACKPROPS_FOLDER, str(zoom), '*', '*.txt'))

  model = load_model(model_file)
  generator = model.layers[0]
  generator.compile(loss='mean_squared_error', optimizer=SGD(lr=75))

  for filname in input_files:
    print "Generating..."
    output_filename = filname.replace(BACKPROPS_FOLDER, FORWARDPROPS_FOLDER) + '.png'
    folder, _ = os.path.split(output_filename)
    mkdir_p(folder)

    inp = np.loadtxt(filname)
    image = forwardprop_single_image(generator, inp)
    image = ((image + 1)*128).astype('uint8')
    misc.imsave(output_filename, image)
