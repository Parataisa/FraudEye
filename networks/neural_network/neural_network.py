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
from evaluation import evaluate, compute_batch_metrics

class NeuralNetworkTrainer:
    def __init__(self, model, optimizer, criterion, device, scheduler, interval_metric=1, batch_interval=10):
        self.net = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.interval_metric = interval_metric
        self.batch_interval = batch_interval

    def train(self, train_loader, val_loader=None, num_epochs=10):
        self.net.to(self.device)

        epoch_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        batch_metrics = []

        patience = 10
        best_f1 = 0  
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.net.train()
            train_loss = 0.0
            correct = 0
            total = 0

            with tqdm(train_loader, unit="batch") as tepoch:
                for batch_idx, (inputs, labels) in enumerate(tepoch, start=1):
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

                    if batch_idx % self.batch_interval == 0:
                        batch_metric = compute_batch_metrics(self.net, inputs, labels, self.device)
                        batch_metric['loss'] = loss.item()
                        batch_metrics.append(batch_metric)

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

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    patience_counter = 0  
                else:
                    patience_counter += 1  

                if patience_counter >= patience:
                    print("Early stopping due to no improvement in F1 score.")
                    break
            self.scheduler.step(train_loss)

        return epoch_metrics, batch_metrics
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

    net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS, DROP_OUT_RATE)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = NeuralNetworkTrainer(net, optimizer, criterion, device, scheduler, interval_metric=1, batch_interval=1000)
    epoch_metrics, batch_metrics = trainer.train(train_loader, val_loader, NUM_EPOCHS)

    params = {
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'output_size': OUTPUT_SIZE,
        'num_hidden_layers': NUM_HIDDEN_LAYERS,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'drop_out_rate' : DROP_OUT_RATE
    }

    trainer.save_model('./data/models/neural_network_model', {'epoch_metrics': epoch_metrics, 'batch_metrics': batch_metrics}, params)
    
    plot_metrics({'epoch_metrics': epoch_metrics, 'batch_metrics': batch_metrics})

def load_and_evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './data/models/neural_network_model_lr_0.001_epochs_40_input_30_hidden_64_layers_3_dropout_0.4_output_1.pth'

    checkpoint = torch.load(model_path)
    net = Net(checkpoint['params']['input_size'], 
              checkpoint['params']['hidden_size'], 
              checkpoint['params']['output_size'], 
              checkpoint['params']['num_hidden_layers'],
              checkpoint['params']['drop_out_rate'])
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)

    epoch_metrics = checkpoint['metrics']['epoch_metrics']
    batch_metrics = checkpoint['metrics']['batch_metrics']
    params = checkpoint['params']

    print(f"Neural Network Metrics:\n\t Accuracy: {epoch_metrics['accuracy']},\n\t Precision: {epoch_metrics['precision']},\n\t Recall: {epoch_metrics['recall']},\n\t F1 Score: {epoch_metrics['f1']},\n\t ROC-AUC: {epoch_metrics['roc_auc']}")
    print(f"Parameters:\n\t Input Size: {params['input_size']},\n\t Hidden Size: {params['hidden_size']},\n\t Output Size: {params['output_size']},\n\t Number of Hidden Layers: {params['num_hidden_layers']},\n\t Learning Rate: {params['learning_rate']},\n\t Number of Epochs: {params['num_epochs']}")

    plot_metrics({'epoch_metrics': epoch_metrics, 'batch_metrics': batch_metrics})


if __name__ == "__main__":
    # Uncomment the function you want to run
    #train_model()
    load_and_evaluate_model()
