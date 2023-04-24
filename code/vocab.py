import jieba
import pandas as pd
import torchtext
import constants
import re
from collections import Counter
from nltk.corpus import stopwords


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

if __name__ == "__main__":
    print("hello world")
    vocab = build_vocab()
    print(vocab)
