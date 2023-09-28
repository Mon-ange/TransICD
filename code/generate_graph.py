import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_module.translate_icd import translate_icd
from data import prepare_datasets
from predictor import Predictor

class ICDMap:

    array = None
    G = None
    def __init__(self, node_num):
        self.array = np.zeros((node_num, node_num))
        H = nx.path_graph(node_num)
        self.G = nx.Graph()
        self.G.add_nodes_from(H)

    def add_edge(self, node1, node2):
        self.array[node1][node2] = 1
        self.array[node2][node1] = 1
        self.G.add_edge(node1, node2)
        self.G.add_edge(node2, node1)

    def print_map(self):
        for i in range(len(self.array)):
            print(self.array[i])

    def draw_map(self):
        nx.draw(self.G, with_labels=True, edge_color='b', node_color='g')
        plt.show()


def generate_graph(model, device):
    predictor = Predictor(model, device)
    train_set, dev_set, test_set, train_labels, train_label_freq, input_indexer, mlb = prepare_datasets(
        data_setting='full', batch_size=8, max_len=1500)
    dataLoader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=1)
    all = 0
    right = 1
    print('mlb size: ')
    print(mlb.transform('J98.414')[0])
    icdMap = ICDMap(len(mlb.transform('J98.414')[0]))
    for batch in dataLoader:
        output = predictor.predict(text=batch['text'])
        for index in range(len(batch)):
            all = all + 1
            predict_icd = translate_icd(mlb, output[index])
            actual = translate_icd(mlb, batch['codes'][index])
            predict_index = np.nonzero(np.array(output[index]))[0][0]
            actual_index = np.nonzero(np.array(batch['codes'][index]))[0][0]
            if actual == predict_icd:
                right = right + 1
            elif actual != predict_icd:
                icdMap.add_edge(predict_index, actual_index)
        break
    precise = float(right) / float(all)
    print(f'precise: {precise}')
    icdMap.print_map()
    icdMap.draw_map()

if __name__ == "__main__":
    # output = [[1,2,3,4,5],[5,4,3,2,1]]
    # result = np.argmax(output, axis=1)
    # mlb = MultiLabelBinarizer()
    # mlb.fit_transform([('1','2','5')])
    # mat = np.array([0,0,1])
    # mat = mat.reshape(1,-1)
    # print(mat)
    # print(mlb.inverse_transform(mat))
    # predict = [0,0,0,1,0,0,0]
    # print(np.nonzero(np.array(predict))[0][0])
    icdMap = ICDMap(30)
    icdMap.add_edge(20,10)
    icdMap.draw_map()