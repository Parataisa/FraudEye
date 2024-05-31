import os
import sys 
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data.get_data import get_data

# Define the 1D CNN model
class CNN1D(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, 1)
        self.conv2 = nn.Conv1d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * (input_size - 4) // 2, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

x_train, y_train = get_data()

# Standardize the Data 
scaler = StandardScaler()
X = scaler.fit_transform(x_train)

# Inspect the shape of the data
print(f"Original data shape: {X.shape}")

# Determine the input dimensions
n_samples, n_features = X.shape
input_size = n_features  # Use the number of features directly for 1D CNN

# Reshape the data for 1D CNN
X = X.reshape((X.shape[0], 1, input_size))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.3, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

device = torch.device('cpu')  # Use CPU instead of GPU
model = CNN1D(input_size=input_size, output_size=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
batch_size = 32
num_batches = X_train_tensor.size(0) // batch_size

for epoch in range(epochs):
    model.train()
    for i in range(num_batches):
        X_batch = X_train_tensor[i*batch_size:(i+1)*batch_size]
        y_batch = y_train_tensor[i*batch_size:(i+1)*batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred = torch.max(outputs.data, 1)
    y_pred = y_pred.numpy()
    y_true = y_test_tensor.numpy()

accuracy = accuracy_score(y_true, y_pred)
confusion = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(confusion)
print('Classification Report:')
print(report)
