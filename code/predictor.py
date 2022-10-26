import numpy as np


class Predictor:

    # model -预测模型
    # device - 跑模型的设备，建议使用GPU
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, text):
        self.model.eval()
        text = text.to(self.device)
        probabs, _, attn_weights = self.model(text)
        return probabs
