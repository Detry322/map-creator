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

from app import OUTPUT_TILES_FOLDER, BACKPROPS_FOLDER
from app.utils import mkdir_p

NOISE_SIZE = 100

def backprop_single_image(generator, image):
  i = Input(shape=(100, 1))
  local = LocallyConnected1D(1, 1, use_bias=False)
  l = local(i)
  r = Reshape((100,))(l)
  output = generator(r)
  model = Model(inputs=i, outputs=output)
  model.compile(loss='mean_squared_error', optimizer=SGD(lr=75))
  X = np.array([[[1.0]]*NOISE_SIZE])
  Y = np.array([image])
  model.fit(X, Y, epochs=200)
  return local.get_weights()


def backprop(model_file, zoom):
  input_files = glob.glob(os.path.join(OUTPUT_TILES_FOLDER, str(zoom), '*', '*.png'))

  model = load_model(model_file)
  generator = model.layers[0]
  generator.trainable = False
  for layer in generator.layers:
    layer.trainable = False

  for filname in input_files:
    output_filename = filname.replace(OUTPUT_TILES_FOLDER, BACKPROPS_FOLDER) + '.txt'
    folder, _ = os.path.split(output_filename)
    mkdir_p(folder)

    image = misc.imread(filname, mode='RGB') / 127.5 - 1
    weights = backprop_single_image(generator, image)
    np.savetxt(output_filename, weights)
