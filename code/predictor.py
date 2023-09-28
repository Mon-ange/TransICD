import numpy as np
import torch.nn.functional as F

class Predictor:

    # model -预测模型
    # device - 跑模型的设备，建议使用GPU
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, text):
        self.model.eval()
        text = text.to(self.device)
        output, _, attn_weights = self.model(text)
        probabs = F.softmax(output).detach().cpu().tolist()
        print(f'probabs: {probabs}')
        index = np.argmax(probabs, axis=1)
        preds = np.zeros_like(probabs)
        for i in range(len(probabs)):
            preds[i, index[i]] = 1
        return preds
