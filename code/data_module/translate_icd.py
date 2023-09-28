import numpy as np


def translate_icd(mlb, tensor):
    mat = np.array(tensor).reshape(1, -1)
    return mlb.inverse_transform(mat)[0][0]