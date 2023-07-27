import multiprocessing
import numpy as np
import pandas as pd
from create_vocab import segment_sentence
from gensim.models import Word2Vec

def get_w2v_model():
    num_cores = multiprocessing.cpu_count()
    min_count = 0
    window = 5
    num_negatives = 5
    embed_size = 128
    print('Training CBOW embedding...')
    return Word2Vec(min_count=min_count, window=window, vector_size=embed_size, negative=num_negatives, workers=num_cores - 1)


def embed_words(input_file_path, out_filename):
    input_file_df = pd.read_csv(input_file_path)
    print("Segment sentences...")
    sentences = []
    for index, text in enumerate(input_file_df['TEXT']):
        if(index % 100 == 0 and index != 0):
            print(f'{index} out of total {len(input_file_df["TEXT"])}')
        sentences.append(segment_sentence(text))
    #TODO: code descripition
    #
    w2v_model = get_w2v_model()
    w2v_model.build_vocab(sentences, progress_per = 10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.init_sims(replace=True)
    w2v_model.save(f'../../data/hainan/{out_filename}')

def map_vocab_to_embed(vocab_filepath, embed_filename, out_filepath):
    model = Word2Vec.load(embed_filename)
    wv = model.wv
    del model
    embed_size = len(wv.word_vec(wv.index_to_key[0]))
    word_to_index = {}
    with open(vocab_filepath,'r',encoding='utf8') as fin, open(out_filepath,'w',encoding='utf8') as fout:
        pad_embed = np.zeros(embed_size)
        unk_embed = np.random.randn(embed_size)
        unk_embed_normalized = unk_embed / float(np.linalg.norm(unk_embed + 1e-6))
        fout.write('<PAD>' + ' ' + np.array2string(pad_embed, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
        fout.write('<UNK>' + ' ' + np.array2string(unk_embed_normalized, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
        word_to_index['<PAD>'] = 0
        word_to_index['<UNK>'] = 1
        for line in fin:
            word = line.strip()
            word_embed = wv.word_vec(word)
            fout.write(word + ' ' + np.array2string(word_embed, max_line_width=np.inf, separator=' ')[1:-1] + '\n')
            word_to_index[word] = len(word_to_index)
    return word_to_index

if __name__ == '__main__':
    # embed_words(f'../../data/hainan/train_full.csv','vocab_ik.w2v')
    map_vocab_to_embed('../../data/hainan/vocab_ik.csv', '../../data/hainan/vocab_ik.w2v', '../../data/hainan/vocab_ik.embed')
