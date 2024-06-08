import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.get_data import get_data
from networks.neural_network.model import Net
from networks.neural_network.hyperparameter import *

class NeuralNetworkTrainer:
    def __init__(self, net, optimizer, criterion, device, scheduler=None):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def train(self, train_loader, val_loader=None, num_epochs=10):
        self.net.to(self.device)

        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            self.net.train()
            train_loss = 0.0
            correct = 0
            total = 0

            with tqdm(train_loader, unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    inputs = inputs.float().to(self.device)
                    labels = labels.float().to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.net(inputs).squeeze()
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    predictions = torch.round(torch.sigmoid(outputs))
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    accuracy = correct / total

                    tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

            train_loss /= len(train_loader.dataset)
            train_acc = correct / total

            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}")

                # Implement early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('./data/models/neural_network_model.pth')

                # Step the scheduler without epoch parameter
                if self.scheduler:
                    self.scheduler.step(val_loss)
            else:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Monitor gradients and weight updates
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
                    weight_norm = param.data.norm().item()
                    if grad_norm > 0.0001:
                        print(f"Param name: {name}, Grad norm: {grad_norm:.4f}, Weight norm: {weight_norm:.4f}")

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def evaluate(self, val_loader):
        self.net.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float().to(self.device)
                labels = labels.float().to(self.device)
                outputs = self.net(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                predictions = torch.round(torch.sigmoid(outputs))
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        return val_loss, val_acc


def load_data(batch_size=BATCH_SIZE):
    X_train, Y_train = get_data()
    X_train_data, X_test, Y_train_data, Y_test = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, stratify=Y_train)
    X_train_data, X_val, Y_train_data, Y_val = train_test_split(X_train_data, Y_train_data, test_size=0.2, shuffle=True, stratify=Y_train_data)
    X_train_data = np.array(X_train_data)
    Y_train_data = np.array(Y_train_data)

    X_train_resampled, y_train_resampled = oversample_data(X_train_data, Y_train_data)

    train_dataset = TensorDataset(torch.from_numpy(X_train_resampled).float(), torch.from_numpy(y_train_resampled).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val.values).float(), torch.from_numpy(Y_val.values).float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, train_val_loader, X_test, Y_test


def get_trained_Model():
    model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
    if(not os.path.exists('./data/models/neural_network_model.pth')):
        print("Model not found")
        return None
    model.load_state_dict(torch.load('./data/models/neural_network_model.pth'))
    model.eval()
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
    print(f"Logistic Regression Metrics:\n\t Accuracy: {accuracy_log_reg},\n\t Precision: {precision_log_reg},\n\t Recall: {recall_log_reg},\n\t F1 Score: {f1_log_reg},\n\t ROC-AUC: {roc_auc_log_reg}\n")

def oversample_data(X_train_data, Y_train_data, fraud_percentage=0.2):
    fraud_indices = np.where(Y_train_data == 1)[0]
    non_fraud_indices = np.where(Y_train_data == 0)[0]

    num_fraud = int(fraud_percentage * len(non_fraud_indices) / (1 - fraud_percentage))
    oversample_fraud_indices = np.random.choice(fraud_indices, size=num_fraud, replace=True)
    oversample_indices = np.concatenate([non_fraud_indices, oversample_fraud_indices])

    X_train_resampled = X_train_data[oversample_indices]
    y_train_resampled = Y_train_data[oversample_indices]

    return X_train_resampled, y_train_resampled


def main():
    train_loader, train_val_loader, X_test, Y_test = load_data()

    net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
    
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = NeuralNetworkTrainer(net, optimizer, criterion, device, scheduler)
    trainer.train(train_loader, train_val_loader, NUM_EPOCHS)
    trainer.save_model('./data/models/neural_network_model.pth')

    model = get_trained_Model()
    evaluate(model, X_test, Y_test)

if __name__ == "__main__":
    main()