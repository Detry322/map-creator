import os, glob
import numpy as np
from scipy import misc

from app import INPUT_TILES_FOLDER

def images_from_filesystem(zoom):
  glob_path = os.path.join(INPUT_TILES_FOLDER, str(zoom), '*', '*.png')
  files = glob.glob(glob_path)
  files.sort()
  while True:
    for filename in files:
      try:
        yield misc.imread(filename)
      except IOError:
        pass
