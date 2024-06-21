import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import Net
from hyperparameter import *
from dataHandler import DataHandler
from visualization import plot_metrics
from evaluation import evaluate 

class NeuralNetworkTrainer:
    def __init__(self, model, optimizer, criterion, device, scheduler, interval_metric=1):
        self.net = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.interval_metric = interval_metric

    def train(self, train_loader, val_loader=None, num_epochs=10):
        self.net.to(self.device)
        epoch_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        patience = 15
        best_roc_auc = 0  
        patience_counter = 0

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
                if val_loader is not None and epoch % self.interval_metric == 0:
                    metrics = self.evaluate(val_loader)
                    epoch_metrics['accuracy'].append(metrics['accuracy'])
                    epoch_metrics['precision'].append(metrics['precision'])
                    epoch_metrics['recall'].append(metrics['recall'])
                    epoch_metrics['f1'].append(metrics['f1'])
                    epoch_metrics['roc_auc'].append(metrics['roc_auc'])
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {accuracy:.4f}")

                    if metrics['roc_auc'] > best_roc_auc:
                        best_roc_auc = metrics['roc_auc']
                        patience_counter = 0  
                    else:
                        patience_counter += 1  

                    if patience_counter >= patience:
                        print("Early stopping due to no improvement in roc_auc score.")
                        break

            self.scheduler.step(train_loss)

        return {'epoch_metrics': epoch_metrics}

    def save_model(self, file_path, metrics, params):
        hyperparameters_str = f"lr_{params['learning_rate']}_epochs_{params['num_epochs']}_input_{params['input_size']}_hidden_{params['hidden_size']}_layers_{params['num_hidden_layers']}_dropout_{params['drop_out_rate']}_output_{params['output_size']}"
        file_path = f"{file_path}_{hyperparameters_str}.pth"
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'params': params
        }, file_path)

    def evaluate(self, val_loader):
        metrics = evaluate(self.net, val_loader, self.device)
        return metrics



def train_model():
    train_loader, val_loader = DataHandler.load_data(BATCH_SIZE)

    net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS, dropout_rate=DROP_OUT_RATE)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = NeuralNetworkTrainer(net, optimizer, criterion, device, scheduler, interval_metric=1)
    metrics = trainer.train(train_loader, val_loader, NUM_EPOCHS)

    params = {
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'output_size': OUTPUT_SIZE,
        'num_hidden_layers': NUM_HIDDEN_LAYERS,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'drop_out_rate' : DROP_OUT_RATE
    }

    trainer.save_model('./data/models/neural_network_model', metrics, params)
    
    plot_metrics(metrics, None, None)

def load_and_evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './data/models/neural_network_model_lr_0.0001_epochs_40_input_30_hidden_256_layers_3_dropout_0.0_output_1.pth'

    checkpoint = torch.load(model_path)
    net = Net(checkpoint['params']['input_size'], 
              checkpoint['params']['hidden_size'], 
              checkpoint['params']['output_size'], 
              checkpoint['params']['num_hidden_layers'],
              checkpoint['params']['drop_out_rate'])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    _, test_loader = DataHandler.load_data(BATCH_SIZE)

    test_metrics = evaluate(net, test_loader, device)

    true_labels_test = test_metrics['true_labels']
    predicted_labels_test = test_metrics['predicted_labels']

    plot_metrics({'epoch_metrics': checkpoint['metrics']['epoch_metrics']}, true_labels_test, predicted_labels_test)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, ROC AUC: {test_metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    # Uncomment the function you want to run
    #train_model()
    load_and_evaluate_model()
