import numpy as np
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.get_data import get_data
from networks.neural_network.model import Net
from networks.neural_network.hyperparameter import *

class DataHandler: 
    @staticmethod
    def load_data(batch_size=64):
        X_train, Y_train = get_data()
        
        X_train_data, X_test, Y_train_data, Y_test = train_test_split(
            X_train, Y_train, test_size=0.2, shuffle=True, stratify=Y_train, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_data = scaler.fit_transform(X_train_data)
        X_test = scaler.transform(X_test)
        
        X_train_resampled, y_train_resampled = DataHandler.oversample_data(
            np.array(X_train_data), np.array(Y_train_data), fraud_percentage=0.5
        )
        
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_resampled).float(), torch.from_numpy(y_train_resampled).int()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(Y_test.values).int() 
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    @staticmethod
    def oversample_data(X_train_data, Y_train_data, fraud_percentage):
        fraud_indices = np.where(Y_train_data == 1)[0]
        non_fraud_indices = np.where(Y_train_data == 0)[0]

        num_fraud = int(fraud_percentage * len(non_fraud_indices) / (1 - fraud_percentage))

        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(X_train_data[fraud_indices])
        neighbors = knn.kneighbors(return_distance=False)

        oversample_fraud_indices = []
        for i in range(num_fraud):
            fraud_idx = np.random.choice(len(fraud_indices))
            actual_fraud_idx = fraud_indices[fraud_idx]
            neighbor_idx = neighbors[fraud_idx][np.random.choice(5)]
            alpha = np.random.random()
            synthetic_sample = (1 - alpha) * X_train_data[actual_fraud_idx] + alpha * X_train_data[neighbor_idx]
            oversample_fraud_indices.append(synthetic_sample)

        oversample_fraud_indices = np.array(oversample_fraud_indices)
        X_train_resampled = np.concatenate([X_train_data, oversample_fraud_indices])
        y_train_resampled = np.concatenate([Y_train_data, np.ones(len(oversample_fraud_indices))])

        return X_train_resampled, y_train_resampled
    
    @staticmethod
    def get_trained_model():
        model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
        model_path = './data/models/neural_network_model.pth'
        
        if not os.path.exists(model_path):
            print("Model not found")
            return None
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model