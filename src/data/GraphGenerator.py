import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import data.translate_icd
from data.data import prepare_datasets
from predictors.Predictor import Predictor

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
    right = 0
    icdMap = ICDMap(len(mlb.transform('J98.414')[0]))
    for batch in dataLoader:
        output = predictor.predict(text=batch['text'])
        for index in range(len(batch)):
            all = all + 1
            predict_icd = translate_icd(mlb, output[index])
            actual = translate_icd(mlb, batch['codes'][index])
            predict_index = np.nonzero(np.array(output[index]))[0][0]
            actual_index = np.nonzero(np.array(batch['codes'][index]))[0][0]
            if predict_index == actual_index:
                right = right + 1
            elif predict_index != actual_index:
                icdMap.add_edge(predict_index, actual_index)
    precise = float(right) / float(all)
    print(f'precise: {precise}')
    icdMap.print_map()
    icdMap.draw_map()

if __name__ == "__main__":
    icdMap = ICDMap(30)
    icdMap.add_edge(20,10)
    icdMap.draw_map()