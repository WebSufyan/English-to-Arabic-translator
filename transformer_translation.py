#import the neccessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import re

from transformers import BertTokenizer , XLMTokenizer, AutoTokenizer

major_tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
#set up the tokenizers from the awesome pre-trained models of huggingface
eng_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
ara_tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic')

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''################################ preparing data for the model #################################
################################################################################################ '''

# setting up tokenizer and fields for our proccessed data
pad_index = eng_tokenizer.convert_tokens_to_ids(eng_tokenizer.pad_token)
ara_pad_index = ara_tokenizer.convert_tokens_to_ids(ara_tokenizer.pad_token)

engli = Field(use_vocab=False, tokenize=eng_tokenizer.encode, pad_token=pad_index)
arabi = Field(use_vocab=False, tokenize=ara_tokenizer.encode, pad_token=pad_index)

fields = {'English': ('en', engli), 'Arabic': ('ara', arabi)}
path = 'C:\\Users\\User\\OneDrive\\Bureau\\coding\\AI projects and website portfolio\\ARTIFICIAL INTELLIGENCE Specials\\english arabic translation'
train_data, test_data = TabularDataset.splits(path,     
                                            train = 'train.json',
                                            test = 'test.json',
                                            format = 'json',
                                            fields = fields)

train_iter, test_iter = BucketIterator.splits((train_data, test_data),
                                                batch_sizes=(8, 8),
                                                # shuffle=False,
                                                # sort_key=lambda x: len(x.comment_text),
                                                sort = False,
                                                device = device)

'''##########################################################################################################
#########################################################################################################'''

'''################################# building the transformer model ########################################
#########################################################################################################'''

class transformer(nn.Module):
    def __init__(self, embedding_size, eng_vocab_size, ar_vocab_size, src_pad_idx, num_heads,
                  num_encoder_layers, num_decoder_layer, forward_expansion, max_len, dropout, device):
        super(transformer, self).__init__()
        
        self.eng_word_embedding = nn.Embedding(eng_vocab_size, embedding_size) 
        self.eng_posisional_embedding = nn.Embedding(max_len, embedding_size)
        
        self.arab_word_embedding = nn.Embedding(ar_vocab_size, embedding_size)
        self.arab_posisional_embedding = nn.Embedding(max_len, embedding_size)
        
        self.device = device
        
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layer,
                                          forward_expansion, dropout)
        
        self.fc_out = nn.Linear(embedding_size, ar_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.src_pad_idx = src_pad_idx        
        
    def make_src_mask(self, src):
        source_mask = src.transpose(0, 1) == self.src_pad_idx
        return source_mask.to(self.device)
    
    def forward(self, src, target):
        src_seq_len, N = src.shape
        target_seq_len, N = target.shape
        
        src_positions = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N)).to(self.device)
        target_positions = (torch.arange(0, target_seq_len).unsqueeze(1).expand(target_seq_len, N)).to(self.device)
        
        embed_src = self.dropout(self.eng_word_embedding(src) + self.eng_posisional_embedding(src_positions))
        embed_target = self.dropout(self.arab_word_embedding(target) + self.arab_posisional_embedding(target_positions))
        
        src_padding_mask = self.make_src_mask(src)
        src_mask = self.transformer.generate_square_subsequent_mask(src_seq_len)
        trg_mask = self.transformer.generate_square_subsequent_mask(target_seq_len).to(self.device)
    
        out = self.transformer(embed_src, embed_target, src_key_padding_mask = src_padding_mask, tgt_mask = trg_mask)
        
        out = self.fc_out(out)

        return out

# setting up the hyperparameters of the model
eng_vocab_size = eng_tokenizer.vocab_size
ar_vocab_size = ara_tokenizer.vocab_size
embedding_size = 512
src_pad_idx = 0
num_heads = 4
num_encoder_layers = 2
num_decoder_layer = 2
forward_expansion = 4
max_len = 100
dropout = 0.1

torch.manual_seed(43)

model = transformer(embedding_size, eng_vocab_size, ar_vocab_size,
                    src_pad_idx, num_heads, num_encoder_layers,
                    num_decoder_layer, forward_expansion, max_len, dropout, device).to(device)

#adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00009)

pad_idx = eng_tokenizer.convert_tokens_to_ids(eng_tokenizer.pad_token)
criterion = nn.CrossEntropyLoss(ignore_index= pad_idx)

epochs = 13
losses = []
test_losses = []

# begin training
for epoch in tqdm(range(epochs)):
    
    model.train()
    for idx, batch in enumerate(train_iter):
        
        x_train = batch.en.to(device)
        y_train = batch.ara.to(device)
        
        output = model(x_train, y_train[:-1, :])
        
        output = output.reshape(-1, output.shape[2])
        y_train = y_train[1:].reshape(-1)
        
        optimizer.zero_grad()
        
        loss = criterion(output, y_train)
        
        loss.backward()        
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        
        optimizer.step()
        
    losses.append(loss)
#validating the model 
    with torch.no_grad():
        model.eval()
        for idx2, test_batch in enumerate(test_iter):
            x_test = test_batch.en.to(device)
            y_test = test_batch.ara.to(device)
            
            y_val = model(x_test, y_test[:-1, :])
            y_val = y_val.reshape(-1, y_val.shape[2])
            y_test = y_test[1:].reshape(-1)
            
    test_cost = criterion(y_val, y_test)      
    
    test_losses.append(test_cost)
    
    if epoch % 5 == 0:
        print(f'\ntrain loss: {loss.item():.3f}\ntest loss : {test_cost:.3f}')
        
plt.plot(losses, label = 'train')
plt.plot(test_losses, label = 'validation')
plt.legend()

#testing
# model.load_state_dict(torch.load('arabic_english_translat_80_epochs.pt'))

while True:
    
    eng_test = input('enter sentence: ')

    eng_test_token = eng_tokenizer.encode(eng_test) 
    
    tensor_sent = torch.LongTensor(eng_test_token).unsqueeze(1).to(device)
    outpt = [2]
    # outpt = set(outpt)
    for i in range(50):
        outpt_tensor = torch.LongTensor(outpt).unsqueeze(1).to(device)
        
        with torch.no_grad():
            predict = model(tensor_sent, outpt_tensor)
            
        best_guess = torch.argmax(predict, 2)[-1, :].item()
        outpt.append(best_guess)
        if best_guess == 3:
            break

        jom = ara_tokenizer.decode(outpt)
        
    print(jom[2:])
    
    if eng_test == 'q':
        break



# torch.save(model.state_dict(), 'toarabic_2heads_2layers_20epochs_1024embed.pt')































