''' manual text tokenizing and counting available words in english '''

import pandas as pd
import re
from collections import Counter

from transformers import BertTokenizer , XLMTokenizer, AutoTokenizer

major_tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
#set up the tokenizers from the awesome pre-trained models of huggingface
eng_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
ara_tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic')

filename = 'ara.txt'

file = open(filename, 'rt', encoding="utf8")

text = file.read()

splitted = text.split('\t')

#read the delimited tab sepearated text in csv format
dataframe = pd.read_csv("ara.txt", delimiter="\t", header = None)

#drop third column which is just metadata we don't need
df = dataframe.drop(dataframe.columns[2], 1)
df1 = df.rename(columns = {0: 'English', 1: 'Arabic'})
# drop_rows = df1.drop(df.index[:73], inplace = True)

#seperate english and arabic columns
english = df1['English']
arabic = df1['Arabic']


eng_words = []
for sentence in english:
    tokened = tokenizer(sentence)
    for i in tokened:
        eng_words.append(i)

available_words = set(eng_words)        

ara_words = []
for jomla in arabic_tokenize:
    for kalima in jomla:
        ara_words.append(kalima)

# creating vocab size in english and map unique index for each word
english_unique = Counter(eng_words)
english_unique = sorted(english_unique, key=english_unique.get, reverse=True)
eng_vocab_size = len(english_unique)
eng_word2idx = {eng_word: idx for idx, eng_word in enumerate(english_unique)}

encoded_eng_sentences = [[eng_word2idx[word] for word in sentence] for sentence in split_eng]
new_encoded_eng_sentences = [i for i in encoded_eng_sentences if (len(i) > 1)]
splitted_eng_encoded = encoded_eng_sentences[1234:]
# eng_shuffled = random.shuffle()

# creating vocab size in arabic and map index for each word
arabic_unique = Counter(ara_words)
arabic_unique = sorted(arabic_unique, key=arabic_unique.get, reverse=True)
arab_vocab_size = len(arabic_unique)
ara_word2idx = {ara_word: idx2 for idx2, ara_word in enumerate(arabic_unique)}

encoded_arab_sentences = [[ara_word2idx[kalma] for kalma in jomla] for jomla in split_arab_reverse]
new_encoded_arab_sentences = [i for i in encoded_arab_sentences if (len(i) > 1)]
splitted_arab_encoded = encoded_arab_sentences[1234:] 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
