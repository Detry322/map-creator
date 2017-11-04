import numpy as np 
from app import INPUT_TILES_FOLDER, OUTPUT_TILES_FOLDER


functions = {
  'greyscale': 0,
  'resize': 1,
}

def preprocess_tiles(zoom, *args):
  for arg in args:
    if arg in functions:
      pass

  input_files = glob.glob(os.path.join(INPUT_TILES_FOLDER, str(zoom), '*', '*.png'))  
  output_files = glob.glob(os.path.join(OUTPUT_TILES_FOLDER, str(zoom), '*', '*.png'))  
  
  for i in xrange(input_files):
  	f = input_files[i]
  	img = misc.imread(f)
  	img = greyscale(img)
  	img = resize(img, p) #TODO replace p
  	misc.imsave(output_files[i], img)

def greyscale(img):
  return img.mean(axis=2)

def resize(img, *args):
  p = args[0]
  return misc.imresize(img, p)
