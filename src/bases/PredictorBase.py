class PredictorBase(object):

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self):
        raise NotImplementedError
