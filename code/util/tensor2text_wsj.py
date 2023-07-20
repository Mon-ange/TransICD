import csv
import codecs
import json

csvFile = codecs.open('../../data/hainan/test_debug.csv', "r", 'utf-8')
reader = csv.reader(csvFile)

csvIFile = codecs.open('../../data/hainan/vocab_zh.csv', "r", 'utf-8')
vocab_reader = csv.reader(csvIFile)

csvWFile = codecs.open('tensor2text_wsj.csv', "w", 'utf-8')
writer = csv.writer(csvWFile)

# load vocab to Dictionary
dictionary = {}
vocab_index = 2
dictionary[0] = ' '
for vocab in vocab_reader:
    dictionary[vocab_index] = vocab
    vocab_index += 1

for item in reader:
    if reader.line_num == 1:
        continue
    translated_text = []
    text_tensors = json.loads(item[1])
    for tensor in text_tensors:
        translated_text.append(dictionary[tensor])
    writer.writerow([item[0], translated_text, item[2], item[3]])
