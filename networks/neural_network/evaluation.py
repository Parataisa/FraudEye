import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
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

def evaluate(model, test_loader, device, BATCH_SIZE=32):
    X_test = []
    Y_test = []
    for inputs, labels in test_loader:
        X_test.extend(inputs.cpu().numpy())
        Y_test.extend(labels.cpu().numpy())
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    y_pred_log_reg = predict(model, X_test, device, BATCH_SIZE)
    accuracy = accuracy_score(Y_test, y_pred_log_reg)
    precision = precision_score(Y_test, y_pred_log_reg)
    recall = recall_score(Y_test, y_pred_log_reg)
    f1 = f1_score(Y_test, y_pred_log_reg)
    roc_auc = roc_auc_score(Y_test, y_pred_log_reg)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
    }
    #print(f"Neural Network Metrics:\n\t Accuracy: {accuracy},\n\t Precision: {precision},\n\t Recall: {recall},\n\t F1 Score: {f1},\n\t ROC-AUC: {roc_auc}")
    
    return metrics