import json
import os
import sys
import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import logging
import matplotlib.pyplot as plt
import torch.optim as optim
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import Dataset, DataLoader
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



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device

def load_data():
    datasets = []
    with open(TRAINING_FILE_PATH, 'r') as train_f:
        train_data = json.load(train_f)
    with open(TEST_FILE_PATH, 'r') as test_f:
        test_data = json.load(test_f)

    train_data['users'].sort()
    test_data['users'].sort()
    assert train_data['users'] == test_data['users'], \
        f"Training and testing data users do not match.\n" + \
        f"Train users: {train_data['users']}\nTest users: {test_data['users']}"
    

    all_train_x = []
    all_train_y = []
    all_test_x = []
    all_test_y = []

    for user in train_data['users']:
        train_x = train_data['user_data'][user]['x']
        train_y = train_data['user_data'][user]['y']
        test_x = test_data['user_data'][user]['x']
        test_y = test_data['user_data'][user]['y']

        all_train_x.extend(train_x)
        all_train_y.extend(train_y)
        all_test_x.extend(test_x)
        all_test_y.extend(test_y)

    # Convert lists to numpy arrays
    all_train_x = np.array(all_train_x, dtype=np.int64)
    all_train_y = np.array(all_train_y, dtype=np.int64)
    all_test_x = np.array(all_test_x, dtype=np.int64)
    all_test_y = np.array(all_test_y, dtype=np.int64)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x).to(torch.device("mps")), torch.tensor(train_y).to(torch.device("mps")))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x).to(torch.device("mps")), torch.tensor(test_y).to(torch.device("mps")))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    return train_loader, test_loader, datasets

load_data()