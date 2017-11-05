from app.data import ZoomLoader
from app.models import all_models

def train_model(model_type, zoom_level, model_file=None):
  ModelClass = all_models[model_type]
  loader = ZoomLoader(zoom_level)
  image_size = loader.random().shape
  model = ModelClass(image_size, loader, model_file=model_file)
  model.train()
