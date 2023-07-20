import csv
import codecs
csvFile = codecs.open('../../data/hainan/dataset_triage.csv', "r", 'utf-8')
reader = csv.reader(csvFile)

csvWFile = codecs.open("../../data/hainan/dataset.csv", "w", 'utf-8')
writer = csv.writer(csvWFile)

for item in reader:
    if reader.line_num == 1:
        continue
    if len(item[1]) < 10 or item[2] == '':
        continue
    writer.writerow([item[0], item[1], item[2]])


csvWFile.close()
csvFile.close()
