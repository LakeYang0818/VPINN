import matplotlib.pyplot as plt

from .tools import output_dir

"""Plots the loss over time"""


def plot_loss(loss_tracker: dict, *, show: bool = False):
    # Plot the boundary loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_b'], label=r'boundary loss', color='darkred')

    # Plot the variational loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_v'], label=r'variational loss', color='navy')

    # Add the grid
    plt.grid()

    # Set labels and titles
    plt.title('Loss over time')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'Total loss', rotation=90)
    plt.yscale('log')
    plt.legend(shadow=True, loc='upper right', fontsize=18, ncol=1)
    if show:
        plt.show()
    else:
        plt.savefig('Results/' + output_dir + '/loss.pdf')
        plt.close()
