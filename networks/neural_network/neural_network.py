import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from networks.neural_network.model import Net
from networks.neural_network.hyperparameter import *
from networks.neural_network.dataHandler import DataHandler
from networks.neural_network.visualization import plot_metrics
from networks.neural_network.evaluation import evaluate

class NeuralNetworkTrainer:
    def __init__(self, model, optimizer, criterion, device, interval_metric=1):
        self.net = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
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
                    self.optimizer.step()

                    train_loss += loss.item()
                    predictions = torch.round(torch.sigmoid(outputs))
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    accuracy = correct / total

                    tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

                train_loss /= len(train_loader)
                if val_loader is not None:
                    metrics = self.evaluate(val_loader)
                    accuracy_metric.append(metrics['accuracy'])
                    precision.append(metrics['precision'])
                    recall.append(metrics['recall'])
                    f1.append(metrics['f1'])
                    roc_auc.append(metrics['roc_auc'])

                if epoch % self.interval_metric == 0:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {accuracy:.4f}")

        metrics = {
            'accuracy': accuracy_metric,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
        }

        return metrics

    def save_model(self, file_path, metrics):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'params': {
                'input_size': INPUT_SIZE,
                'hidden_size': HIDDEN_SIZE,
                'output_size': OUTPUT_SIZE,
                'num_hidden_layers': NUM_HIDDEN_LAYERS,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS
            }
        }, file_path)

    def evaluate(self, val_loader):
        self.net.eval()
        metrics = evaluate(self.net, val_loader, self.device)
        return metrics


def train_model():
    train_loader, val_loader = DataHandler.load_data(BATCH_SIZE)

    net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = NeuralNetworkTrainer(net, optimizer, criterion, device)
    metrics = trainer.train(train_loader, val_loader, NUM_EPOCHS)
    trainer.save_model('./data/models/neural_network_model.pth', metrics)
    plot_metrics(metrics)

def load_and_evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load('./data/models/neural_network_model.pth')
    net = Net(checkpoint['params']['input_size'], 
              checkpoint['params']['hidden_size'], 
              checkpoint['params']['output_size'], 
              checkpoint['params']['num_hidden_layers'])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    metrics = checkpoint['metrics']
    params = checkpoint['params']

    print(f"Neural Network Metrics:\n\t Accuracy: {metrics['accuracy']},\n\t Precision: {metrics['precision']},\n\t Recall: {metrics['recall']},\n\t F1 Score: {metrics['f1']},\n\t ROC-AUC: {metrics['roc_auc']}")
    print(f"Parameters:\n\t Input Size: {params['input_size']},\n\t Hidden Size: {params['hidden_size']},\n\t Output Size: {params['output_size']},\n\t Number of Hidden Layers: {params['num_hidden_layers']},\n\t Learning Rate: {params['learning_rate']},\n\t Number of Epochs: {params['num_epochs']}")

    plot_metrics(metrics)


if __name__ == "__main__":
    # Uncomment the function you want to run
    train_model()
    load_and_evaluate_model()