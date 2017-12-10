import os, glob
import numpy as np
from scipy import misc

from app import INPUT_TILES_FOLDER, OUTPUT_TILES_FOLDER
from app.utils import mkdir_p

def greyscale(img, *args):
  return img.mean(axis=2)

def resize(img, *args):
  p = float(args[0]) if len(args) > 0 else .5
  return misc.imresize(img, p, interp='lanczos')

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

def write_processed_image(processed_image, input_filename):
  output_filename = input_filename.replace(INPUT_TILES_FOLDER, OUTPUT_TILES_FOLDER)
  folder, _ = os.path.split(output_filename)
  mkdir_p(folder)
  with open(output_filename, 'w+') as f:
    misc.imsave(f, processed_image)

def preprocess_tile(functions, input_filename):
  try:
    img = misc.imread(input_filename, mode='RGB')
    for func in functions:
      img = func(img)
    write_processed_image(img, input_filename)
  except:
    pass

def preprocess_tiles(zoom, *args):
  functions = parse_args(args)
  input_files = glob.glob(os.path.join(INPUT_TILES_FOLDER, str(zoom), '*', '*.png'))

  for input_filename in input_files:
    preprocess_tile(functions, input_filename)
