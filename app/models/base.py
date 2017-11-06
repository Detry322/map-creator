class BaseModel(object):
  def __init__(self, image_size, image_loader, model_file=None):
    self.model = None
    self.image_size = image_size
    self.image_loader = image_loader

    if model_file is not None:
      self._construct_from_file(model_file)
    else:
      self._construct_model()

  def _construct_from_file(self, filename):
    raise NotImplementedError("This was never overridden.")

  def _construct_model(self, image_size):
    raise NotImplementedError("This was never overridden.")

  def train(self):
    raise NotImplementedError("This was never overridden.")

  def generate_image(self):
    raise NotImplementedError("This was never overridden.")
