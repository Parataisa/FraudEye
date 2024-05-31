import pandas as pd
import numpy as np
import os
import sys
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.get_data import get_data
from networks.neural_network.model import Net
from networks.neural_network.hyperparameter import *

class NeuralNetworkTrainer:
    def __init__(self, net, optimizer, criterion, device):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_loader, num_epochs=NUM_EPOCHS):
        self.net.to(self.device)
        self.net.train()
        ## https://adamoudad.github.io/posts/progress_bar_with_tqdm/ 
        for epoch in range(1, num_epochs + 1):
            with tqdm(train_loader, unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    inputs = inputs.float().to(self.device)
                    labels = labels.float().to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.net(inputs).squeeze()
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    predictions = outputs.argmax(dim=0, keepdim=True)
                    correct = predictions.eq(labels).sum().item()
                    accuracy = correct / len(labels == 1)

                    tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

        self.save_model('./data/models/neural_network_model.pth')

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)


def load_data(batch_size=BATCH_SIZE):
    X_train, y_train = get_data()
    print(X_train.shape, y_train.shape)
    X_train_data, X_test, Y_train_data, Y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train, random_state=42)

    X_train_data = np.array(X_train_data)
    Y_train_data = np.array(Y_train_data).reshape(-1, 1)
    print((y_train == 1).sum().item() / y_train.shape[0])


    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_data, Y_train_data)

    fraud_data = X_train_resampled[y_train_resampled == 1]
    non_fraud_data = X_train_resampled[y_train_resampled == 0]
    print(f"Fraud data: {fraud_data.shape}, Non-fraud data: {non_fraud_data.shape}")

    tensor1 = torch.from_numpy(X_train_resampled)
    tensor2 = torch.from_numpy(y_train_resampled)

    train_dataset = TensorDataset(tensor1, tensor2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Data loaded")
    return train_loader, X_test, Y_test

def get_trained_Model():
    print("Setting up model")
    model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
    print("Model set up")
    if(not os.path.exists('./data/models/neural_network_model.pth')):
        print("Model not found")
        return None
    model.load_state_dict(torch.load('./data/models/neural_network_model.pth'))
    model.eval()
    print("Model loaded")
    return model

def predict(model, data):
    model.eval()
    data_tensor = torch.tensor(data.values).float()
    logits = model.forward(x=data_tensor)
    output = torch.sigmoid(logits).detach().numpy() > THRESHOLD
    return output

def evaluate(model, X_test, Y_test):
    y_pred_log_reg = predict(model, X_test)
    accuracy_log_reg = accuracy_score(Y_test, y_pred_log_reg)
    precision_log_reg = precision_score(Y_test, y_pred_log_reg)
    recall_log_reg = recall_score(Y_test, y_pred_log_reg)
    f1_log_reg = f1_score(Y_test, y_pred_log_reg)
    roc_auc_log_reg = roc_auc_score(Y_test, y_pred_log_reg)
    print(f"Logistic Regression Metrics: Accuracy: {accuracy_log_reg}, Precision: {precision_log_reg}, Recall: {recall_log_reg}, F1 Score: {f1_log_reg}, ROC-AUC: {roc_auc_log_reg}")


def main():
    train_loader, X_test, Y_test = load_data()

    net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
    
    #Be careful, use BCE for binary, CrossEntropy for Multi-class
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = NeuralNetworkTrainer(net, optimizer, criterion, device)
    print("Training started")
    trainer.train(train_loader, NUM_EPOCHS) 
    print("Training completed")
    trainer.save_model('./data/models/neural_network_model.pth')
    print("Model saved")

    model = get_trained_Model()
    evaluate(model, X_test, Y_test)

if __name__ == "__main__":
    main()