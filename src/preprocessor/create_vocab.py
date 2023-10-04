import warnings

import pandas as pd
import requests
import torchtext.vocab
from requests.auth import HTTPBasicAuth
import json

from collections import Counter

warnings.filterwarnings('ignore')

train_file = '../../data/hainan/train_full.csv'
output_file_path = '../../data/hainan/vocab_ik.csv'

es_server_url = 'https://127.0.0.1:9200/_analyze'

def read_train_discharge(path):
    discharge_df = pd.read_csv(train_file, encoding='utf8')
    return discharge_df

def segment_sentence(sentence):
    response = requests.post(es_server_url,
                             json = {
                                 "text": sentence,
                                 "analyzer": "ik_smart"
                             },
                             auth = HTTPBasicAuth("elastic", "wkM_c38vp8tcIx7Qq=E5"),
                             verify = False)
    data = response.json()
    result = []
    for item in data["tokens"]:
        if item["type"] == "CN_WORD" or item["type"] == "CN_CHAR":
            result.append(item["token"])
    return result

def add_data_to_dictionary(counter, data):
    print('Adding data to dictionary...')
    for index, sentence in enumerate(data):
        if(index % 100 == 0):
            print(f'{index} out of total {len(data)}')
        sentence_array = segment_sentence(sentence)
        counter.update(sentence_array)

def write_dictionary(counter):
    vocab = torchtext.vocab.vocab(counter)
    with open(output_file_path, 'w+', encoding='utf8') as fout:
        for str in vocab.get_itos():
            fout.write(f'{str}\n')


def create_dictionary_by_IK():
    # 初始化最初的字典，可以将分好的词加入Counter中
    counter = Counter()
    discharge_df = read_train_discharge(train_file)
    add_data_to_dictionary(counter = counter, data = discharge_df['TEXT'])
    write_dictionary(counter)


if __name__ == '__main__':
    create_dictionary_by_IK()
