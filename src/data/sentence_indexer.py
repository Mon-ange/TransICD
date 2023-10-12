from utils import Indexer
from constants import *
from data.sentence_segmentor import SentenceSegmentor


class SentenceIndexer(object):

    def __init__(self):
        self.indexer = Indexer()
        self.indexer.add_and_get_index(PAD_SYMBOL)
        self.indexer.add_and_get_index(UNK_SYMBOL)
        with open(VOCAB_FILE_PATH, 'r', encoding='utf8') as fin:
            for line in fin:
                word = line.strip()
                self.indexer.add_and_get_index(word)
        self.oov_words = 0

    def index(self, segmented_sentence, max_len):
        text_len = max_len if len(segmented_sentence) > max_len else len(segmented_sentence)
        sentence_indexed = [self.indexer.index_of(PAD_SYMBOL)] * max_len
        print(segmented_sentence)
        for i in range(text_len):
            if self.indexer.index_of(segmented_sentence[i]) >= 0:
                sentence_indexed[i] = self.indexer.index_of(segmented_sentence[i])
            else:
                self.oov_words += 1
                sentence_indexed[i] = self.indexer.index_of(UNK_SYMBOL)
        return sentence_indexed


sentence_indexer = SentenceIndexer()
