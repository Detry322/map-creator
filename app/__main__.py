import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy
numpy.random.seed(234421)

from app.download import download_tiles, prune_tiles
from app.data import ZoomLoader
from app.models import BasicDCGAN, Autoencoder

def get_args():
  parser = argparse.ArgumentParser(description="map-creator uses DCGANs to generate pictures of map tiles")
  parser.add_argument('--action', help="Defines the action to be taken",
                      default='train', choices=['download', 'train', 'generate'])
  return parser.parse_args()

def main():
  args = get_args()
  if args.action == 'download':
    print "Downloading tiles..."
    download_tiles()
    print "Pruning tiles..."
    prune_tiles()
  elif args.action == 'train':
    loader = ZoomLoader(15)
    model = BasicDCGAN(loader)
    model.train()
  else:
    raise Exception("Sorry, haven't implemented that yet!")

if __name__ == "__main__":
  main()
