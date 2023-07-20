import csv
import codecs
import json

csvFile = codecs.open('tensor2text.csv', "r", 'utf-8')
reader = csv.reader(csvFile)

csvWFile = codecs.open('probs2max.csv', "w", 'utf-8')
writer = csv.writer(csvWFile)

for item in reader:
    if reader.line_num == 1:
        continue

    pred = json.loads(item[2])
    target = json.loads(item[3])
    maxp = 0
    maxpi = -1
    for i in range(0,len(pred)):
        if pred[i] > maxp:
            maxp = pred[i]
            maxpi = i

    maxt = 0
    maxtj = -1
    for j in range(0, len(target)):
        if target[j] > maxt:
            maxt = target[j]
            maxtj = j

    writer.writerow([item[0], item[1], maxpi, maxp, maxtj, maxt])
