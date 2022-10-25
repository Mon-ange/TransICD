# TransICD: Transformer Based Code-wise Attention Model for Explainable ICD Coding



## Data Preparations

### 1 Download dataset from mimic official website

[[MIMIC-III Clinical Database Demo v1.4 (physionet.org)](https://physionet.org/content/mimiciii-demo/1.4/)](https://mimic.mit.edu/docs/iii/)

Download the dataset using wget

```shell
wget -r -N -c -np https://physionet.org/files/mimiciii-demo/1.4/
```

Copy the required data files,

* NOTEEVENTS.csv
* DIAGNOSES_ICD.csv
* PROCEDURES_ICD.csv
* D_ICD_DIAGNOSES.csv
* D_ICD_PROCEDURES.csv
* ICD9_descriptions (Already given)

into the folder,

### 2 Initialize virtual environment

### 3 Download Stop Word

Download stopwords from nltk official website, http://www.nltk.org/nltk_data/ into directory - `TransICD\nltk_data\corpora`

### 3 Run data preprocess

```shell
python preprocessor.py
```



## Data Preprocess

### 1 discharge full

**Method:** preprocessor.write_discharge_summaries()

**Source file:** NOTEEVENTS.csv

**Target File:**  disch_full.csv

It will scan the TEXT column of each line from NOTEEVENTS.csv, then clean the text into a summary version which is handled using stemmer and stopwords. Stemmer is a library called Snow Ball Stemmer. Each word in the text will be translated with stemmer. The word both in the text and stopwords will be removed from the original text, as the word in stopwords library can be considered meaningless.

write_discharge_summaries will return two parameters, which are hadm_id_set and disch_full_filename. hadm_id_set is a set which contains all the id representing each patient's stay.

### 2 discharge full word to vector

**Method:** preprocessor.embed_words()

**Source file:** disch_full.csv

**Target file:** disch_full.w2v

### 3 vocab

**Method:** preprocessor.build_vocab

**Source file:** train_full.csv

**Target file:** vocab.csv

### 4 vocab

**Method:** preprocessor.map_vocab_to_embed

**Source file:** vocab.csv, disch_full.w2v

**Target file:** vocab.embed

### 5 All codes filtered

**Method**: preprocessor.combine_diag_proc_codes()

**Source file:** DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv

**Target File:** ALL_CODES_filtered.csv

### 6 notes labeld

**Method:** preprocessor.combine_notes_codes

**Source file:** ALL_CODES_filtered.csv, disch_full.csv

**Target file:** notes_labeled.csv

### 7 test, train, dev

**Method:** split_data

**Source file:** notes_labeled.csv

**Target file:** train_50.csv ,dev_50.csv ,test_50.csv



## Tain TransICD model

