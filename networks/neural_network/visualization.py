import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_metrics(metrics, Y_test, y_pred):
    epoch_metrics = metrics.get('epoch_metrics', {})
    metrics_keys = list(epoch_metrics.keys())

    if len(metrics_keys) > 1:
        metrics_keys = metrics_keys[1:]

    num_metrics = len(metrics_keys)

    num_columns = 2
    num_rows = math.ceil(num_metrics / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 4 * num_rows))
    axes = axes.flatten()

    metric_idx = 0

    num_epochs = len(epoch_metrics[metrics_keys[0]])
    interval_metric = max(1, num_epochs // 10)
    epochs = list(range(interval_metric, num_epochs + 1, interval_metric))
    for metric in metrics_keys:
        ax = axes[metric_idx]
        ax.plot(epochs, epoch_metrics[metric][:len(epochs)], label=metric)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
        metric_idx += 1

    for i in range(metric_idx, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if Y_test and y_pred:
        conf_matrix = confusion_matrix(Y_test, y_pred)

        fig, ax = plt.subplots()
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        for (i, j), val in np.ndenumerate(conf_matrix):
            ax.text(j, i, f'{val}', ha='center', va='center', color='red')
        plt.title("Confusion Matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()
    else:
        print("Confusion matrix is empty, skipping the plot.")

    plt.show()
