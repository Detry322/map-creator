import os, glob
import numpy as np
from scipy import misc

from app import INPUT_TILES_FOLDER, OUTPUT_TILES_FOLDER

def greyscale(img, *args):
  return img.mean(axis=2)

def resize(img, *args):
  p = float(args[0]) if len(args) > 0 else .5
  return misc.imresize(img, p)

all_functions = {
  'greyscale': greyscale,
  'resize': resize,
}

def make_func(func, f_args):
  return lambda x: func(x, *f_args)

def parse_args(args):
  functions = []
  func = None
  f_args = []
  for arg in args:
    if arg in all_functions:
      if func:
        functions.append(make_func(func, f_args))
      func = all_functions[arg]
      f_args = []
    else:
      f_args.append(arg)
  functions.append(make_func(func, f_args))
  return functions

def preprocess_tiles(zoom, *args):
  functions = parse_args(args)

  input_files = glob.glob(os.path.join(INPUT_TILES_FOLDER, str(zoom), '*', '*.png'))

  for i in xrange(len(input_files)):
    f = input_files[i]
    output = os.path.join(OUTPUT_TILES_FOLDER, f[len(INPUT_TILES_FOLDER):])
    open(output, 'w+').close()
    img = misc.imread(f)
    for func in functions:
      img = func(img)
    misc.imsave(output, img)
