import pandas as pd 
import os 
from sklearn.metrics import roc_auc_score 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

path = "./data/mlproject22" if os.path.exists("./data/mlproject22") else "."
train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
X_train = train_data.drop(columns = "Class")
y_train = train_data["Class"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Split the data into training and testing sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_split, y_train_split)

# Predictions
y_pred_log_reg = log_reg.predict(X_test_split)

# Evaluation
print("Logistic Regression Classification Report:")
print(classification_report(y_test_split, y_pred_log_reg))
print("Confusion Matrix:")
print(confusion_matrix(y_test_split, y_pred_log_reg))

# Metrics
accuracy_log_reg = accuracy_score(y_test_split, y_pred_log_reg)
precision_log_reg = precision_score(y_test_split, y_pred_log_reg)
recall_log_reg = recall_score(y_test_split, y_pred_log_reg)
f1_log_reg = f1_score(y_test_split, y_pred_log_reg)
roc_auc_log_reg = roc_auc_score(y_test_split, y_pred_log_reg)

print(f"Logistic Regression Metrics: Accuracy: {accuracy_log_reg}, Precision: {precision_log_reg}, Recall: {recall_log_reg}, F1 Score: {f1_log_reg}, ROC-AUC: {roc_auc_log_reg}")

# Save the Logistic Regression model
joblib.dump(log_reg, 'logistic_regression_model.pkl')