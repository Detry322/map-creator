import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
np.random.seed(234421)

from app.data import ZoomLoader
from app.download import download_tiles, prune_tiles
from app.preprocess import preprocess_tiles
from app.train import train_model
from app.generate import generate_tiles
from app.backprop import backprop
from app.forwardprop import forwardprop
from app.random import random

def get_args():
  parser = argparse.ArgumentParser(description='map-creator uses DCGANs to generate pictures of map tiles')
  parser.add_argument('--zoom', help='Zoom size', type=str, default='13')
  parser.add_argument('--download', help='Download tiles', action='store_true')
  parser.add_argument('--preprocess', help='Preprocessing args', nargs='+')
  parser.add_argument('--backprop', help='backprop', action='store_true')
  parser.add_argument('--random', help='random', action='store_true')
  parser.add_argument('--forwardprop', help='forwardprop', action='store_true')
  parser.add_argument('--train', help='Train tiles', action='store_true')
  parser.add_argument('--model_type', help='The model to train/generate with', type=str, default='BestDCGAN')
  parser.add_argument('--model_file', help='The h5 model file', type=str)
  parser.add_argument('--generate', help='Generating tiles', action='store_true')
  return parser.parse_args()

def main(): 
  args = get_args()
  if args.download:
    print 'Downloading tiles...'
    download_tiles(args.zoom)
    print 'Pruning tiles...'
    prune_tiles()
  if args.preprocess:
    print 'Preprocessing tiles...'
    preprocess_tiles(args.zoom, *args.preprocess)
  if args.train:
    print "Training model..."
    train_model(args.model_type, args.zoom, model_file=args.model_file)
  if args.generate:
    print "Generating tiles until Control-C'd..."
    generate_tiles(args.model_type, args.model_file)
  if args.backprop:
    print "Backproping until Control-C'd..."
    backprop(args.model_file, args.zoom)
  if args.forwardprop:
    print "Forwardproping until Control-C'd..."
    forwardprop(args.model_file, args.zoom)
  if args.random:
    print "Doing random stuff"
    random(args.model_file)

if __name__ == '__main__':
  main()
