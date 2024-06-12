import matplotlib.pyplot as plt

def plot_metrics(metrics):
    num_epochs = len(metrics['accuracy'])
    interval_metric = max(1, num_epochs // 10) 

    epochs = list(range(interval_metric, num_epochs + 1, interval_metric))

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot(epochs, metrics[metric][:len(epochs)], label=metric) 
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    plt.show()
