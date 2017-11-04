import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy
numpy.random.seed(234421)

from app.data import ZoomLoader
from app.download import download_tiles, prune_tiles
from app.preprocess import preprocess_tiles
from app.models import BasicDCGAN, Autoencoder

def get_args():
  parser = argparse.ArgumentParser(description='map-creator uses DCGANs to generate pictures of map tiles')
  parser.add_argument('--download', help='Zoom size for downloading tiles',
                      nargs='?', type=int, const=15)
  parser.add_argument('--preprocess', help='Args for which types of preprocessing',
                      nargs='*')
  parser.add_argument('--train', help='Zoom size for training tiles',
                      nargs='?', type=int, const=15)
  parser.add_argument('--generate', help='Generating tiles')
  return parser.parse_args()

def main(): 
  args = get_args()
  if args.download:
    print 'Downloading tiles...'
    download_tiles(args.download)
    print 'Pruning tiles...'
    prune_tiles()
  if args.preprocess:
    print 'Preprocessing tiles...'
    preprocess_tiles(*args.preprocess)
  elif args.train:
    loader = ZoomLoader(args.train)
    model = BasicDCGAN(loader)
    model.train()
  else:
    raise Exception('Sorry, haven\'t implemented that yet!')

if __name__ == '__main__':
  main()
