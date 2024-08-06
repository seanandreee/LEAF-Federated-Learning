import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
import string
from collections import defaultdict
from utils.language_utils import letter_to_vec, word_to_indices
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from torch.nn.functional import one_hot
from utils.language_utils import letter_to_vec, word_to_indices
print('starting script')

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

TRAINING_FILE_PATH = os.path.join('data', 'shakespeare', 'data', 'train', 'all_data_niid_0_keep_0_train_9.json')
TEST_FILE_PATH = os.path.join('data', 'shakespeare', 'data', 'test', 'all_data_niid_0_keep_0_test_9.json')
print('file path confirmed')


'''
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load training and test data
training_data = load_data(TRAINING_FILE_PATH)
test_data = load_data(TEST_FILE_PATH)

print(len(test_data))
print(len(training_data))

'''
#limit data usage
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

print(len(TRAINING_FILE_PATH))
print(len(TEST_FILE_PATH))


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
print('function pass')
#Load and preprocess data - Tokenize data
# Load and preprocess the training and test data
preprocessed_train_data = load_and_preprocess(TRAINING_FILE_PATH)
preprocessed_test_data = load_and_preprocess(TEST_FILE_PATH)

print(type(preprocessed_test_data))
print(type(preprocessed_train_data))
print(f"Number of users in preprocessed training data: {len(preprocessed_train_data['users'])}")
print(f"Number of users in preprocessed test data: {len(preprocessed_test_data['users'])}")

print('data preprocessed')
# Tokenize the preprocessed data and build vocabulary
tokenized_train_data, vocab = tokenize_data(preprocessed_train_data)
tokenized_test_data, _ = tokenize_data(preprocessed_test_data)
print(f"Vocabulary size: {len(vocab)}")
print(f"Example tokenized data (training): {tokenized_train_data['user_data'][tokenized_train_data['users'][0]]['x'][:3]}")
print(type(tokenized_test_data))
print(type(tokenized_train_data))
print('data tokenized')

#debugging statement below
'''for user in tokenized_train_data['users'][:5]:
    for text in tokenized_train_data['user_data'][user]['x'][:3]:
        print(f"Tokenized text train length: {len(text)}")
        print(text)
'''

#create sequences
def create_sequences_and_targets(tokenized_data, seq_length=3, max_users = 2):
    sequences = []
    targets = []

    for user in tokenized_data['users'][:max_users]:
        for text in tokenized_data['user_data'][user]['x']:
            if len(text) < seq_length + 1:
                continue
            for i in range(seq_length, len(text)):
                seq = text[i-seq_length:i]
                target = text[i]
                sequences.append(seq)
                targets.append(target)
    
    return sequences, targets

def train_and_evaluate_model(epochs=50, seq_length=10, batch_size=32, embedding_dim=128, hidden_dim=256, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    # Load and preprocess the training and test data
    preprocessed_train_data = load_and_preprocess(TRAINING_FILE_PATH)
    preprocessed_test_data = load_and_preprocess(TEST_FILE_PATH)
    print(f"Number of users in preprocessed training data: {len(preprocessed_train_data['users'])}")
    print(f"Number of users in preprocessed test data: {len(preprocessed_test_data['users'])}")

    # Tokenize the preprocessed data and build vocabulary
    tokenized_train_data, vocab = tokenize_data(preprocessed_train_data)
    tokenized_test_data, _ = tokenize_data(preprocessed_test_data)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Example tokenized data (training): {tokenized_train_data['user_data'][tokenized_train_data['users'][0]]['x'][:3]}")

    # Create sequences and targets
    train_sequences, train_targets = create_sequences_and_targets(tokenized_train_data, seq_length=seq_length)
    test_sequences, test_targets = create_sequences_and_targets(tokenized_test_data, seq_length=seq_length)
    print(f"Number of training sequences: {len(train_sequences)}")
    print(f"Number of training targets: {len(train_targets)}")
    print(f"Number of test sequences: {len(test_sequences)}")
    print(f"Number of test targets: {len(test_targets)}")

    # Convert sequences and targets to tensors
    train_sequences_tensor = torch.tensor(train_sequences, dtype=torch.long)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.long)
    test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.long)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_sequences_tensor, train_targets_tensor)
    test_dataset = TensorDataset(test_sequences_tensor, test_targets_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Define LSTM
    class My_LSTM(nn.Module):
        def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim):
            super(My_LSTM, self).__init__()
            self.embedding = nn.Embedding(input_dim, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = x.to(self.embedding.weight.device)
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            lstm_out = self.dropout(lstm_out)
            output = self.fc(lstm_out[:, -1, :])
            return output

    # Instantiate the model
    input_dim = len(vocab)
    output_dim = len(vocab)
    model = My_LSTM(input_dim, output_dim, embedding_dim, hidden_dim).to(device)
    print("Model instantiated successfully.")

    # Training loop
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def calculate_topk_accuracy(model, data_loader, k=3):
        model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)

                # Get top-k predictions
                _, predicted_indices = output.topk(k, dim=1)

                correct_predictions += torch.any(predicted_indices == batch_y.unsqueeze(1), dim=1).sum().item()
                total_predictions += batch_y.size(0)

        accuracy = correct_predictions / total_predictions
        return accuracy

    all_accuracies = []
    all_losses = []

    for epoch in range(epochs):   
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        all_losses.append(epoch_loss)

        if epoch % 5 == 0:
            accuracy = calculate_topk_accuracy(model, train_loader)
            print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}, Train K-Accuracy: {accuracy * 100:.2f}%')
            all_accuracies.append(accuracy)

    # test eval
    accuracy = calculate_topk_accuracy(model, test_loader)
    print(f'Test K-Accuracy: {accuracy * 100:.2f}%')

# Call the training function
train_and_evaluate_model()