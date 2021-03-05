import pandas as pd
import re

from sklearn.model_selection import train_test_split

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

english = english.apply(lambda x: re.sub(r'[^a-zA-Z.!?]+', r' ', x))
arabic = arabic.apply(lambda x: re.sub(r',', r' ', x))
    
dataset = pd.concat([english, arabic], axis = 1)

train, test = train_test_split(dataset, test_size = 0.05, shuffle = True, random_state = 43)

train.to_json('train.json', orient = 'records', lines=True)
test.to_json('test.json', orient = 'records', lines=True)


''' proccessing using nltk and spacy '''
# from nltk.tokenize import word_tokenize, sent_tokenize
# import spacy

# #splitting
# # train, test = train_test_split(df1, test_size=0.2, shuffle = True, random_state = 101)
# # testing, validation = train_test_split(test, test_size=0.5, shuffle = True, random_state = 101)

# #saving
# # train.to_csv('train.csv', index = False)
# # testing.to_csv('test.csv', index = False)
# # validation.to_csv('validation.csv', index = False)

# # train.to_json('train.json', orient = 'records', lines=True)
# # testing.to_json('test.json', orient = 'records', lines=True)
# # validation.to_json('validation.json', orient = 'records', lines=True)
# df1.to_json('eng_ara_translation.json', orient = 'records', lines = True)

# english_tokenize = [word_tokenize(sentence) for sentence in english]
# arabic_tokenize = [word_tokenize(sentence) for sentence in arabic]

# # spacy_ger = spacy.load("en_core_web_sm")
# # spacy_eng = spacy.load("en_core_web_sm")

# def tokenize_eng(text):
#     return word_tokenize(text)
# def tokenize_ara(text):
#     return word_tokenize(text)

# def bert_tokenizer_eng(text):
#     huggingface = tokenizer(text)
#     return huggingface['input_ids']
# def bert_tokenizer_ara(text):
#     huggingface = tokenizer(text)
#     return huggingface['input_ids']


# def tokenize_angalazi(text):
#     return [tok.text for tok in spacy_ger.tokenizer(text)]
# def tokenize_arab(text):
#     return [tok.text for tok in spacy_eng.tokenizer(text)]

# # split_eng = english_tokenize[1234:]
# # split_arab = arabic_tokenize[1234:]























