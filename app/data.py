import os, glob
import numpy as np
import random
from scipy import misc

from app import OUTPUT_TILES_FOLDER

class ZoomLoader(object):
  def __init__(self, zoom):
    self.files = glob.glob(os.path.join(OUTPUT_TILES_FOLDER, str(zoom), '*', '*.png'))
    self.i = -1

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.files)

  def random(self):
    return misc.imread(random.choice(self.files))

  def next(self):
    self.i += 1
    self.i %= len(self)
    return misc.imread(self.files[self.i])
