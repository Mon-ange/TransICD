import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self, config, type1=None, type2=None):
        super(type1, type2).__init__()
        self.config = config
        self.model = None

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("[ERROR] Model haven't been built when save")
        print("[INFO] Save Model")
        self.model.save_weights(checkpoint_path)
        print("[INFO] Model Saved")

    def load(self, checkpoint_path):
        """
        加载checkpoint, 路径定义于配置文件中
        """
        if self.model is None:
            raise Exception("[Exception] You have to build the model first.")

        print("[INFO] Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("[INFO] Model loaded")

    def forward(self):
        raise NotImplementedError