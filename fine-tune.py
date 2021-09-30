# Borrowed and refactored from https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
'''
Build a sentiment analysis model, out of tweets that are labeled as 
non-complaint is a '1' and complaint is a '0'
''' 
from __future__ import print_function
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from utils import text_preprocessing, set_seed, get_device
from model import BertClassifier, initialize_model, preprocessing_for_bert
import torch.nn as nn
from prepare_data import prepare_data

from train_eval import train, bert_predict

device = get_device()  # GPU or cpu

# if your data is not downloaded, uncomment steps below to download and unzip.
# # Download data
# import requests
# request = requests.get("https://drive.google.com/uc?export=download&id=1wHt8PsMLsfX5yNSqrt2fSTcb8LEiclcf")
# with open("data.zip", "wb") as file:
#     file.write(request.content)

# # Unzip data
# import zipfile
# with zipfile.ZipFile('data.zip') as zip:
#     zip.extractall('data')

X, y, X_train, X_val, y_train, y_val, train_val_all, test_data = prepare_data()
# Display 5 samples from the test data
print("Displaying test samples...")
print(test_data.sample(5))

# BERT Tokenizer
from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)  # since model is uncased

# Concatenate train data and test data
all_tweets = np.concatenate([train_val_all.tweet.values, test_data.tweet.values])

# Encode our concatenated data
encoded_tweets = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in all_tweets]

# Find the maximum length
max_len = max([len(sentence) for sentence in encoded_tweets])
print('Max length: ', max_len)

# Now let's tokenize our data.
MAX_LEN = 64

# Print sentence 0 and its encoded token ids
token_ids = list(preprocessing_for_bert(tokenizer, [X[0]], MAX_LEN)[0].squeeze().numpy())
print('Original: ', X[0])
print('Token IDs: ', token_ids)

# Run function `preprocessing_for_bert` on the train set and the validation set
print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_bert(tokenizer, X_train, MAX_LEN)
val_inputs, val_masks = preprocessing_for_bert(tokenizer, X_val, MAX_LEN)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

# For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
batch_size = 32

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# Specify loss function
loss_fn = nn.CrossEntropyLoss()

set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(device, train_dataloader, epochs=2)
train(bert_classifier, loss_fn, optimizer, scheduler, device,
    train_dataloader, val_dataloader, epochs=4, evaluation=True)
    
# Compute predicted probabilities on the test set
probs = bert_predict(bert_classifier, device, val_dataloader)

# TODO Evaluate the Bert classifier evaluate_roc(probs, y_val)

# Run `preprocessing_for_bert` on the test set
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(tokenizer, test_data.tweet, MAX_LEN)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)
# Run `preprocessing_for_bert` on the test set
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_bert(tokenizer, test_data.tweet, MAX_LEN)

# Compute predicted probabilities on the test set
probs = bert_predict(bert_classifier, device, test_dataloader)

# Get predictions from the probabilities
threshold = 0.9
preds = np.where(probs[:, 1] > threshold, 1, 0)

# Number of tweets predicted non-negative
print("Number of tweets predicted non-negative: ", preds.sum())


outputs = test_data[preds==1]
# for _ in range(20):
#     print(outputs.sample(1).tweet) 
print(list(outputs.sample(20).tweet), sep='\n')
print(outputs.sample(20).tweet, sep='\n')