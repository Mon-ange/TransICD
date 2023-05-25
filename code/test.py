import numpy as np


if __name__ == "__main__":
    input = np.random.randint(100,size=4)
    print(input)
    indices = np.argmax(input)
    result = np.zeros_like(input)
    result[indices] = 1
    print(result)

