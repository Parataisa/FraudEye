import numpy as np
import os
import sys

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.get_data import get_data
from networks.neural_network.model import Net
from networks.neural_network.hyperparameter import *

class DataHandler: 
    def load_data(batch_size=None):
        X_train, Y_train = get_data()
        X_train_data, X_test, Y_train_data, Y_test = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, stratify=Y_train, random_state=42)

        X_train_resampled, y_train_resampled = DataHandler.oversample_data(np.array(X_train_data), np.array(Y_train_data), fraud_percentage=0.5)

        train_dataset = TensorDataset(torch.from_numpy(X_train_resampled).float(), torch.from_numpy(y_train_resampled).float())
        test_dataset = TensorDataset(torch.from_numpy(X_test.values).float(), torch.from_numpy(Y_test.values).float())

        if batch_size is None:
            batch_size = len(train_dataset)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader
    
    def oversample_data(X_train_data, Y_train_data, fraud_percentage):
        fraud_indices = np.where(Y_train_data == 1)[0]
        non_fraud_indices = np.where(Y_train_data == 0)[0]

        num_fraud = int(fraud_percentage * len(non_fraud_indices) / (1 - fraud_percentage))
        oversample_fraud_indices = np.random.choice(fraud_indices, size=num_fraud, replace=True)
        oversample_indices = np.concatenate([non_fraud_indices, oversample_fraud_indices])

        X_train_resampled = X_train_data[oversample_indices]
        y_train_resampled = Y_train_data[oversample_indices]

        return X_train_resampled, y_train_resampled
    
    def get_trained_Model():
        model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
        if(not os.path.exists('./data/models/neural_network_model.pth')):
            print("Model not found")
            return None
        model.load_state_dict(torch.load('./data/models/neural_network_model.pth'))
        model.eval()
        return model
