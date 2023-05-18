import csv
import os

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
from collections import defaultdict

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


def load_code_desc(code_desc_filename="icd_description_zh.csv"):
    code_desc_df = pd.read_csv(f'{constants.GENERATED_DIR}/{code_desc_filename}', encoding='utf-8')
    code_text_series = code_desc_df['IN_DIAGNOSIS_CN']
    return code_text_series

def load_code_and_desc(code_desc_filename="icd_description_zh.csv"):
    desc_dict = defaultdict(str)
    desc_df = pd.read_csv(f'{constants.GENERATED_DIR}/{code_desc_filename}', encoding='utf-8')
    # for row in desc_df.iterrows():
    #     os.system("pause")
    #     print('code: ', row['IN_DIAGNOSIS_CODE'],"  desc:",row['IN_DIAGNOSIS_CN'])
    #     desc_dict[row['IN_DIAGNOSIS_CODE'].strip()] = row['IN_DIAGNOSIS_CN'].strip()
    # return desc_dict
    return desc_df



def build_vocab(train_full_filename='dataset_triage.csv', out_filename='vocab_zh.csv'):
    # 导入病案描述文本生成词典
    train_df = pd.read_csv(f'{constants.GENERATED_DIR}/{train_full_filename}', encoding='utf-8')
    full_text_series = train_df['Column2']
    counter = Counter()
    for triage in full_text_series:
        counter.update(tokenizer(triage.strip(),False))
    # 导入ICD编码及其描述生成词典
    code_text_series = load_code_desc()
    for code_text in code_text_series:
        counter.update(tokenizer(code_text.strip(),False))
    vocab = torchtext.vocab.vocab(counter)
    out_file_path = f'{constants.GENERATED_DIR}/{out_filename}'

    with open(out_file_path, 'w', encoding='UTF-8') as fout:
        for str in vocab.get_itos():
            # print(str)
            fout.write(f'{str}\n')


def embed_words(disch_full_filename='dataset_triage.csv', embed_size=128, out_filename='disch_full.w2v'):
    disch_df = pd.read_csv(f'{constants.GENERATED_DIR}/{disch_full_filename}', encoding="utf-8")
    desc_df = load_code_desc()
    sentences = [tokenizer(text.strip(),False) for text in disch_df['Column2']]
    for desc in desc_df:
        sentences.append(tokenizer(desc.strip()))
    num_cores = multiprocessing.cpu_count()
    min_count = 0
    window = 5
    num_negatives = 5
    print(f'Params: embed_size={embed_size}, workers={num_cores-1}, min_count={min_count}, window={window}, negative={num_negatives}')
    w2v_model = Word2Vec(min_count=min_count, window=window, vector_size=embed_size, negative=num_negatives, workers=num_cores-1)
    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.init_sims(replace=True)
    w2v_model.save(f'{constants.GENERATED_DIR}/{out_filename}')
    #logging.info('\n**********************************************\n')
    return out_filename

def vectorize_code_desc(out_filename='code_desc_vectors_zh.csv', vocab_filename='vocab_zh.csv'):
    word_to_idx = {}
    word_to_idx[constants.PAD_SYMBOL] = 0
    word_to_idx[constants.UNK_SYMBOL] = 1
    with open(f'{constants.GENERATED_DIR}/{vocab_filename}', 'r', encoding="utf-8") as fin:
        for line in fin:
            word_to_idx[line.strip()] = len(word_to_idx)
    desc_df = load_code_and_desc()
    with open(f'{constants.GENERATED_DIR}/{out_filename}', 'w') as fout:
        w = csv.writer(fout, delimiter=' ')
        w.writerow(["CODE", "VECTOR"])
        for idx,desc in desc_df.iterrows():
            code = desc['IN_DIAGNOSIS_CODE']
            tokens = tokenizer(desc['IN_DIAGNOSIS_CN'])
            inds = [word_to_idx[t] if t in word_to_idx.keys() else word_to_idx[constants.UNK_SYMBOL] for t in tokens]
            w.writerow([code] + [str(i) for i in inds])

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
    #map_vocab_to_embed()
    vectorize_code_desc()
    #print(vocab)
