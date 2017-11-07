from app.models.basic_dcgan import BasicDCGAN
from app.models.better_dcgan import BetterDCGAN
from app.models.autoencoder import Autoencoder

all_models = {
  'BasicDCGAN': BasicDCGAN,
  'Autoencoder': Autoencoder,
  'BetterDCGAN': BetterDCGAN
}
