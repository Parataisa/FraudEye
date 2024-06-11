import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from networks.neural_network.model import Net
from networks.neural_network.hyperparameter import *
from networks.neural_network.dataHandler import DataHandler
from networks.neural_network.visualization import plot_metrics
from networks.neural_network.evaluation import evaluate

class NeuralNetworkTrainer:
    def __init__(self, net, optimizer, criterion, device, scheduler=None, interval_metric=4):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.interval_metric = interval_metric

    def train(self, train_loader, val_loader=None, num_epochs=10):
        self.net.to(self.device)

        accuracy_metric = []
        precision = []
        recall = []
        f1 = []
        roc_auc = []

        for epoch in range(1, num_epochs + 1):
            self.net.train()
            train_loss = 0.0
            correct = 0
            total = 0

            with tqdm(train_loader, unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    inputs = inputs.float().to(self.device)
                    labels = labels.float().to(self.device).unsqueeze(1)

                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
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
            if epoch % self.interval_metric == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {accuracy:.4f}")

            if val_loader is not None and epoch % self.interval_metric == 0:
                metrics = self.evaluate(val_loader)
                accuracy_metric.append(metrics['accuracy'])
                precision.append(metrics['precision'])
                recall.append(metrics['recall'])
                f1.append(metrics['f1'])
                roc_auc.append(metrics['roc_auc'])

            if self.scheduler:
                self.scheduler.step(train_loss)

        metrics = {
            'accuracy': accuracy_metric,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
        }

        return metrics

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def evaluate(self, val_loader):
        self.net.eval()
        metrics = evaluate(self.net, val_loader, self.device, BATCH_SIZE)
        self.net.train()
        return metrics


def main():
    train_loader, test_loader = DataHandler.load_data(NUM_EPOCHS)

    net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=NUM_EPOCHS//4)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = NeuralNetworkTrainer(net, optimizer, criterion, device)
    metrics = trainer.train(train_loader, test_loader, NUM_EPOCHS)
    trainer.save_model('./data/models/neural_network_model.pth')

    plot_metrics(metrics)
    metrics = evaluate(net, test_loader, device, BATCH_SIZE)
    print(f"Neural Network Metrics:\n\t Accuracy: {metrics['accuracy']},\n\t Precision: {metrics['precision']},\n\t Recall: {metrics['recall']},\n\t F1 Score: {metrics['f1']},\n\t ROC-AUC: {metrics['roc_auc']}")

if __name__ == "__main__":
    main()
