import jieba
import pandas as pd
import torchtext
import constants
import re
from collections import Counter
from nltk.corpus import stopwords
import multiprocessing
from gensim.models import Word2Vec
import numpy as np

my_stopwords = set([stopword for stopword in stopwords.words('chinese')])

def regex_change(line):
    #前缀的正则
    username_regex = re.compile(r"^\d+::")
    #URL，为了防止对中文的过滤，所以使用[a-zA-Z0-9]而不是\w
    url_regex = re.compile(r"""
        (https?://)?
        ([a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)
        (\.[a-zA-Z0-9]+)*
        (/[a-zA-Z0-9]+)*
    """, re.VERBOSE|re.IGNORECASE)
    #剔除日期
    data_regex = re.compile(u"""        #utf-8编码
        年 |
        月 |
        日 |
        (周一) |
        (周二) | 
        (周三) | 
        (周四) | 
        (周五) | 
        (周六)
    """, re.VERBOSE)
    # 剔除英文字符
    decimal_regex = re.compile(r"[^a-zA-Z]\d+")
    # 剔除所有数字
    num_regex = re.compile(r"[+-]?\d+(\.\d*)?")
    #剔除空格
    space_regex = re.compile(r"\s+")

    line = username_regex.sub(r"", line)
    line = url_regex.sub(r"", line)
    line = data_regex.sub(r"", line)
    line = decimal_regex.sub(r"", line)
    line = num_regex.sub(r"", line)
    line = space_regex.sub(r"", line)

    return line
def tokenizer(s,word = False):
    s = regex_change(s)
    if word:
        r = [w for w in s]
    else:
        s = jieba.cut(s, cut_all=False)
        r = " ".join([x for x in s if x not in my_stopwords]).split()
    return r
def build_vocab(train_full_filename='dataset_triage.csv', out_filename='vocab_zh.csv'):
    train_df = pd.read_csv(f'{constants.GENERATED_DIR}/{train_full_filename}')
    full_text_series = train_df['Column2']
    counter = Counter()
    for triage in full_text_series:
        counter.update(tokenizer(triage.strip(),False))
    # print(counter)
    vocab = torchtext.vocab.vocab(counter)
    out_file_path = f'{constants.GENERATED_DIR}/{out_filename}'

    with open(out_file_path, 'w', encoding='UTF-8') as fout:
        for str in vocab.get_itos():
            # print(str)
            fout.write(f'{str}\n')


def embed_words(disch_full_filename='dataset_triage.csv', embed_size=128, out_filename='disch_full.w2v'):
    disch_df = pd.read_csv(f'{constants.GENERATED_DIR}/{disch_full_filename}')
    sentences = [tokenizer(text.strip(),False) for text in disch_df['Column2']]
    # desc_dt = load_code_desc()
    #for desc in desc_dt.values():
    #    sentences.append(clean_text(desc, trantab, my_stopwords, stemmer).split())
    num_cores = multiprocessing.cpu_count()
    min_count = 0
    window = 5
    num_negatives = 5
    #logging.info('\n**********************************************\n')
    #logging.info('Training CBOW embedding...')
    #logging.info(f'Params: embed_size={embed_size}, workers={num_cores-1}, min_count={min_count}, window={window}, negative={num_negatives}')
    print(f'Params: embed_size={embed_size}, workers={num_cores-1}, min_count={min_count}, window={window}, negative={num_negatives}')
    w2v_model = Word2Vec(min_count=min_count, window=window, vector_size=embed_size, negative=num_negatives, workers=num_cores-1)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.init_sims(replace=True)
    w2v_model.save(f'{constants.GENERATED_DIR}/{out_filename}')
    #logging.info('\n**********************************************\n')
    return out_filename

def map_vocab_to_embed(vocab_filename='vocab_zh.csv', embed_filename='disch_full.w2v', out_filename='vocab_zh.embed'):
    model = Word2Vec.load(f'{constants.GENERATED_DIR}/{embed_filename}')
    wv = model.wv
    del model

    embed_size = len(wv.word_vec(wv.index_to_key[0]))
    word_to_idx = {}
    with open(f'{constants.GENERATED_DIR}/{vocab_filename}', 'r', encoding="utf-8") as fin, open(f'{constants.GENERATED_DIR}/{out_filename}', 'w',encoding="utf-8") as fout:
        pad_embed = np.zeros(embed_size)
        unk_embed = np.random.randn(embed_size)
        unk_embed_normalized = unk_embed / float(np.linalg.norm(unk_embed) + 1e-6)
        fout.write(constants.PAD_SYMBOL + ' ' + np.array2string(pad_embed, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
        fout.write(constants.UNK_SYMBOL + ' ' + np.array2string(unk_embed_normalized, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
        word_to_idx[constants.PAD_SYMBOL] = 0
        word_to_idx[constants.UNK_SYMBOL] = 1

        for line in fin:
            word = line.strip()
            word_embed = wv.word_vec(word)
            fout.write(word + ' ' + np.array2string(word_embed, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
            word_to_idx[word] = len(word_to_idx)

    #logging.info(f'Size of training vocabulary (including PAD, UNK): {len(word_to_idx)}')
    return word_to_idx

if __name__ == "__main__":
    print("hello world")
    #vocab = build_vocab()
    #embed_words()
    map_vocab_to_embed()
    #print(vocab)
