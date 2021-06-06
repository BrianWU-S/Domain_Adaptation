import matplotlib.pyplot as plt
import os


def loss_plot(epochs, loss):
    """
        Draw loss curve and save as figure
    """
    num = epochs
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path + 'UNET++_loss.jpg'
    plt.figure()
    plt.plot(x, loss, label='loss')
    plt.legend()
    plt.savefig(save_loss)


def metrics_plot(epochs, name, *args):
    """
        Draw metric value curve and save as figure
    """
    num = epochs
    names = name.split('&')
    metrics_value = args
    i = 0
    x = [i for i in range(num)]
    plot_save_path = r'result/plot/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = plot_save_path + 'UNET++_' + name + '.jpg'
    plt.figure()
    for l in metrics_value:
        plt.plot(x, l, label=str(names[i]))
        i += 1
    plt.legend()
    plt.savefig(save_metrics)