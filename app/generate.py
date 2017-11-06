from app.models import all_models
from app.utils import mkdir_p
from app import GENERATED_TILES_FOLDER

from scipy import misc

import os
import time

def generate_tiles(model_type, model_file):
  ModelClass = all_models[model_type]
  model = ModelClass(None, None, model_file=model_file)

  folder = os.path.join(GENERATED_TILES_FOLDER, '{}'.format(time.time()))
  mkdir_p(folder)

  i = 0
  while True:
    try:
      print "Generating {}...".format(i)
      filename = os.path.join(folder, '{}.png'.format(i))
      misc.imsave(filename, model.generate_image())
      i += 1
    except KeyboardInterrupt:
      exit(0)
