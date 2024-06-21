import pandas as pd
import os
import sys
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# Ensure correct import path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from data.get_data import get_data
from dataHandler import DataHandler

def train_and_evaluate():
    # Load the data
    X_train, Y_train = get_data()
    print(X_train)
    # Split the data into training and testing sets
    X_train_data, X_test, Y_train_data, Y_test = train_test_split(
        X_train, Y_train, test_size=0.2, shuffle=True, stratify=Y_train, random_state=42
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train_fit = scaler.fit_transform(X_train_data)
    X_test = scaler.transform(X_test)


    X_train_resampled, Y_train_resampled = DataHandler.oversample_data(X_train_fit, Y_train_data, fraud_percentage=0.15)
    print(X_train_resampled)
    
    # Initialize the Logistic Regression model
    model = LogisticRegression(random_state=42)

    # Train the model on the training data
    model.fit(X_train_resampled, Y_train_resampled)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Save the Logistic Regression model
    joblib.dump(model, "logistic_regression_model.pkl")
    joblib.dump(scaler, "logistic_regression_scaler.pkl")

    # evaluation
    print("logistic regression classification report:")
    print(classification_report(Y_test, y_pred))
    print("confusion matrix:")
    print(confusion_matrix(Y_test, y_pred))

    # metrics
    accuracy_log_reg = accuracy_score(Y_test, y_pred)
    precision_log_reg = precision_score(Y_test, y_pred, average="weighted")
    recall_log_reg = recall_score(
        Y_test,
        y_pred,
        average="weighted"
    )
    f1_log_reg = f1_score(Y_test, y_pred, average="weighted")
    roc_auc_log_reg = roc_auc_score(Y_test, y_pred, average="weighted")


    print(
        f"logistic regression metrics: accuracy: {accuracy_log_reg}, precision: {precision_log_reg}, recall: {recall_log_reg}, f1 score: {f1_log_reg}, roc-auc: {roc_auc_log_reg}"
    )

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
    roc_auc = roc_auc_score(Y_test, y_pred_proba)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.show()

    # Assuming y_pred is defined

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(Y_test, y_pred)

    # Plot confusion matrix
    plt.figure()
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Non-Fraud", "Fraud"])
    plt.yticks(tick_marks, ["Non-Fraud", "Fraud"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Annotate the confusion matrix
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > conf_matrix.max() / 2.0 else "black")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    train_and_evaluate()
