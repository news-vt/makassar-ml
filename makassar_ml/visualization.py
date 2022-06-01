import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric(history: dict, metric: str) -> plt.Figure:
    """Plots model training and validation metric."""
    fig = plt.figure()
    plt.plot(history[metric], label='train')
    plt.plot(history[f"val_{metric}"], label='val')
    plt.xlim(0, len(history[metric])-1)
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend(loc='upper left')
    return fig