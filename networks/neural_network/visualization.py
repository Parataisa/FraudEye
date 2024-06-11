import matplotlib.pyplot as plt

def plot_metrics(metrics):
    epochs = list(range(1, len(metrics['accuracy']) + 1))

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot(epochs, metrics[metric], label=metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    plt.show()