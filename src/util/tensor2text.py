import csv
import codecs
import json

csvFile = codecs.open('../../data/hainan/test_debug.csv', "r", 'utf-8')
reader = csv.reader(csvFile)

csvIFile = codecs.open('../../data/hainan/vocab_zh.csv', "r", 'utf-8')
Ireader = csv.reader(csvIFile)

csvWFile = codecs.open('tensor2text.csv', "w", 'utf-8')
writer = csv.writer(csvWFile)

index = []
for Iitem in Ireader:
    index.append(Iitem)

for item in reader:
    if reader.line_num == 1:
        continue

    sample_json = item[1]
    data = json.loads(sample_json)

    res = []
    for i in data:
        for ind in range(0, len(index)):
            if ind == i-2:
                res.append(index[ind])
                #删掉text中的0
                break
    writer.writerow([item[0], res, item[2], item[3]])



