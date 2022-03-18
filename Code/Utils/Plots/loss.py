import matplotlib.pyplot as plt

from .tools import output_dir

"""Plots the loss over time"""


def plot_loss(loss_tracker: dict, *, show: bool = False, write_data: bool = True):
    # Plot the boundary loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_b'], label=r'boundary loss', color='darkorange')

    # Plot the variational loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_v'], label=r'variational loss', color='darkslategray')

    # Add the grid
    plt.grid()

    # Set labels and titles
    plt.title('Loss over time')
    plt.xlabel(r'Iteration')
    plt.yscale('log')
    plt.legend(shadow=True, loc='upper right', fontsize=18, ncol=1)
    if show:
        plt.show()
    else:
        plt.savefig('Results/' + output_dir + '/loss.pdf')
        plt.close()

    # Write out the loss data as a csv file
    if write_data:
        import pandas as pd
        df = pd.DataFrame(loss_tracker)
        df.to_csv('Results/' + output_dir + '/loss.csv')

