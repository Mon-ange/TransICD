import numpy as np


if __name__ == "__main__":
    input = np.random.randint(100,size=(3,4))
    print(input)
    indices = np.argmax(input,axis=1)
    print(indices)
    result = np.zeros_like(input)
    print(result)
    print(len(result))
    for i in range(len(result)):
        result[i, indices[i]] = 1
    print(result)

