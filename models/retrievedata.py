import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
import string
from collections import defaultdict
from utils.language_utils import letter_to_vec, word_to_indices
from torchtext.vocab import build_vocab_from_iterator
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

LOSS_FUNCTIONS = {
    "scce": nn.CrossEntropyLoss(), # Sparse categorical cross-entropy.
}
OPTIM_FUNCTIONS = {
    "adam": torch.optim.Adam, # Adam optimizer.
    "sgd": torch.optim.SGD, # Stochastic gradient descent.
}

TRAINING_FILE_PATH = os.path.join('data', 'shakespeare', 'data', 'train', 'all_data_niid_0_keep_0_train_9.json')
TEST_FILE_PATH = os.path.join('data', 'shakespeare', 'data', 'test', 'all_data_niid_0_keep_0_test_9.json')


def load_and_preprocess(file_path):
    def preprocess_data(text):
        # Convert to lowercase
        text = text.lower()
        printable = set(string.printable)
        text = ''.join(filter(lambda x: x in printable, text))
        
        return text
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    preprocessed_data = {'users': [], 'user_data': {}}
    
    for user in data['users']:
        preprocessed_data['users'].append(user)
        preprocessed_data['user_data'][user] = {'x': [], 'y': []}
        
        # Preprocess each piece of text data
        for text in data['user_data'][user]['x']:
            preprocessed_text = preprocess_data(text)
            preprocessed_data['user_data'][user]['x'].append(preprocessed_text)
        
        for text in data['user_data'][user]['y']:
            preprocessed_text = preprocess_data(text)
            preprocessed_data['user_data'][user]['y'].append(preprocessed_text)
    
    return preprocessed_data

# Load and preprocess the training and test data
preprocessed_train_data = load_and_preprocess(TRAINING_FILE_PATH)
preprocessed_test_data = load_and_preprocess(TEST_FILE_PATH)

'''# Print the preprocessed training data
print("Preprocessed Training Data:")
print(json.dumps(preprocessed_train_data, indent=4))

# Print the preprocessed test data
print("Preprocessed Test Data:")
print(json.dumps(preprocessed_test_data, indent=4))

'''
#TOKENIZE DATA
def tokenize_data(data):
    def tokenize(text):
        return list(text) 

    def yield_tokens(data):
        for user in data['users']:
            for text in data['user_data'][user]['x']:
                yield tokenize(text)
            for text in data['user_data'][user]['y']:
                yield tokenize(text)
    
    # Build vocabulary from the tokenized data
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    tokenized_data = {'users': [], 'user_data': {}}
    
    for user in data['users']:
        tokenized_data['users'].append(user)
        tokenized_data['user_data'][user] = {'x': [], 'y': []}
        
        # Tokenize each piece of text data
        for text in data['user_data'][user]['x']:
            tokenized_text = [vocab[token] for token in tokenize(text)]
            tokenized_data['user_data'][user]['x'].append(tokenized_text)
        
        for text in data['user_data'][user]['y']:
            tokenized_text = [vocab[token] for token in tokenize(text)]
            tokenized_data['user_data'][user]['y'].append(tokenized_text)
    
    return tokenized_data, vocab


'''
To-do:
Create sequences
pad sequences
create dataloaders + datasets

'''