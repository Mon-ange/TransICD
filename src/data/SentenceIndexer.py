import warnings
import numpy as np
import requests
import random
from concurrent.futures import ThreadPoolExecutor
from requests.auth import HTTPBasicAuth
import threading
import time
from requests.adapters import HTTPAdapter


class SentenceIndexer:
    es_server_url = ['https://127.0.0.1:9200/_analyze', 'https://127.0.0.1:9201/_analyze', 'https://127.0.0.1:9202/_analyze',
                     'https://127.0.0.1:9203/_analyze']

    def __init__(self):
        self.indexed_sentences = []
        self.completed_count = 0
        self.lock = threading.RLock()
        self.sessions = []
        for url in self.es_server_url:
            session = requests.session()
            adapter = HTTPAdapter(pool_connections=17, pool_block=True)
            session.mount(url, adapter)
            self.sessions.append(session)

    def segment_sentence(self, index, sentence):
        server_num = random.randint(0, 3)
        retry = 0
        while retry < 3:
            try:
                response = self.sessions[server_num].post(self.es_server_url[server_num],
                                                          json={
                                                              "text": sentence,
                                                              "analyzer": "ik_smart"
                                                          },
                                                          auth=HTTPBasicAuth("elastic", "wkM_c38vp8tcIx7Qq=E5"),
                                                          verify=False,
                                                          timeout=10)
                break
            except requests.exceptions.RequestException as e:
                print('retrying...')
                time.sleep(5)
                retry += 1
                if retry == 3:
                    print(e)
                    print("Fail with retry 3 times!")
                    return
        data = response.json()
        result = []
        for item in data["tokens"]:
            if item["type"] == "CN_WORD" or item["type"] == "CN_CHAR":
                result.append(item["token"])
        self.lock.acquire()
        self.indexed_sentences[index] = result
        self.completed_count += 1
        self.lock.release()

    def index(self, sentences):
        warnings.filterwarnings('ignore')
        # print(f'sentence length: {len(sentences)}')
        self.indexed_sentences = np.empty(len(sentences), dtype=list)
        self.completed_count = 0
        print(f'total: {len(sentences)}')
        thread_pool_executor = ThreadPoolExecutor(max_workers=68, thread_name_prefix="test_")
        for index in range(len(sentences)):
            thread_pool_executor.submit(self.segment_sentence, index, sentences[index])
        while self.completed_count != len(sentences):
            time.sleep(2)
        return self.indexed_sentences.tolist()


if __name__ == "__main__":
    sentenceIndexer = SentenceIndexer()
    print(sentenceIndexer.index(["南京市长江大桥", "今天天气不错"]))
    #indexed_sentences = np.empty(len(["南京市长江大桥", "今天天气不错"]))
    #indexed_sentences