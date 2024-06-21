import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def predict(model, data, device, batch_size=32, THRESHOLD=0.5):
    model.eval()
    data_tensor = torch.tensor(data).float()
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_outputs = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            logits = model(inputs)
            outputs = torch.sigmoid(logits)
            all_outputs.extend(outputs.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    predictions = all_outputs > THRESHOLD
    return predictions

def evaluate(model, data_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device).unsqueeze(1)
            outputs = model(inputs)
            predictions = torch.round(torch.sigmoid(outputs))
            true_labels.extend(labels.cpu().numpy().tolist())
            predicted_labels.extend(predictions.cpu().numpy().tolist())

    # Calculate metrics
    metrics = {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(true_labels, predicted_labels, average='weighted'),
        'recall': recall_score(true_labels, predicted_labels, average='weighted'),
        'f1': f1_score(true_labels, predicted_labels, average='weighted'),
        'roc_auc': roc_auc_score(true_labels, predicted_labels, average='weighted')
    }
    return metrics


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    roc_auc = None
    if len(np.unique(y_true)) > 1:  
        roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
    }
    return metrics