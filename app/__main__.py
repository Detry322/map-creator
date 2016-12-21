import argparse

from app.download import download_tiles

def get_args():
  parser = argparse.ArgumentParser(description="map-creator uses DCGANs to generate pictures of map tiles")
  parser.add_argument('--action', help="Defines the action to be taken",
                      default='generate', choices=['download', 'train', 'generate'])
  return parser.parse_args()

def main():
  args = get_args()
  if args.action == 'download':
    download_tiles()
  else:
    raise Exception("Sorry, haven't implemented that yet!")

if __name__ == "__main__":
  main()
