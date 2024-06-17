import matplotlib.pyplot as plt
import math

def plot_metrics(metrics):
    epoch_metrics = metrics.get('epoch_metrics', {})
    batch_metrics = metrics.get('batch_metrics', [])

    num_epoch_metrics = len(epoch_metrics)
    num_batch_metrics = len(batch_metrics[0].keys()) if batch_metrics else 0
    num_metrics = num_epoch_metrics + num_batch_metrics

    num_columns = 5
    num_rows = math.ceil(num_metrics / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 4 * num_rows))
    axes = axes.flatten()  

    metric_idx = 0

    # Plot epoch-level metrics
    if epoch_metrics:
        num_epochs = len(epoch_metrics['accuracy'])
        interval_metric = max(1, num_epochs // 10)
        epochs = list(range(interval_metric, num_epochs + 1, interval_metric))

        for metric in epoch_metrics:
            ax = axes[metric_idx]
            ax.plot(epochs, epoch_metrics[metric][:len(epochs)], label=metric)
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric)
            ax.legend()
            metric_idx += 1

    # Plot batch-level metrics
    if batch_metrics:
        num_batches = len(batch_metrics)
        batch_indices = list(range(1, num_batches + 1))

        for metric in batch_metrics[0].keys():
            y_values = [bm[metric] for bm in batch_metrics if bm[metric] is not None]
            batch_indices = list(range(1, len(y_values) + 1))
            
            ax = axes[metric_idx]
            ax.plot(batch_indices, y_values, label=metric)
            ax.set_xlabel('Batch')
            ax.set_ylabel(metric)
            ax.legend()
            metric_idx += 1

    for i in range(metric_idx, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
