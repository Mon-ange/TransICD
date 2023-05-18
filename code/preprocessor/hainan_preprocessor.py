from collections import Counter

import pandas as pd


def read_csv(file_name):
    return pd.read_csv(file_name)


def split(data_frame):
    train_size = int(len(data_frame) * 0.6)
    test_size = int(len(data_frame) * 0.2)
    dev_size = int(len(data_frame) * 0.2)
    print(train_size)

    train_data = data_frame[:train_size]
    test_data = data_frame[train_size:train_size + test_size]
    dev_data = data_frame[train_size + test_size:]

    train_set = pd.DataFrame({
        "HADM_ID": train_data['Column1'],
        "TEXT": train_data['Column2'],
        "LABELS": train_data['Column4']
    })
    test_set = pd.DataFrame({
        "HADM_ID": test_data['Column1'],
        "TEXT": test_data['Column2'],
        "LABELS": test_data['Column4']
    })
    dev_set = pd.DataFrame({
        "HADM_ID": dev_data['Column1'],
        "TEXT": dev_data['Column2'],
        "LABELS": dev_data['Column4']
    })
    return train_set, test_set, dev_set


def write_dataset(train_set, test_set, dev_set, path):
    train_set.to_csv(path + '/train_full.csv', index=False)
    test_set.to_csv(path + '/test_full.csv', index=False)
    dev_set.to_csv(path + '/dev_full.csv', index=False)


def calculate_codefreq(df):
    counter = Counter()
    for labels in df['Column4'].values:
        for label in str(labels).split(';'):
            counter[label] += 1
    codes, freqs = map(list, zip(*counter.most_common()))
    return pd.DataFrame({'code': codes, 'freq': freqs})


def write_codefreq(df,path):
    df.to_csv(path + '/code_freq.csv', index=False)

if __name__ == '__main__':
    df = read_csv('../../data/hainan/dataset_triage.csv')
    freq_df = calculate_codefreq(df)
    write_codefreq(freq_df,'../../data/hainan')
    train_set, test_set, dev_set = split(df)
    print(train_set)
    print(dev_set)
    write_dataset(train_set, test_set, dev_set, '../../data/hainan')




