import matplotlib.pyplot as plt
import numpy as np
import torch

from .functions import dtest_function, u as u_exact
from .Types.Grid import Grid

"""Plots used in the VPINNS model"""

# Plots the prediction value
def plot_prediction(grid: Grid, y_pred, grid_shape: tuple):

    # 1D plot
    if grid.dim == 1:

        # Plot the exact solution and the model predictions
        plt.plot(grid.data, u_exact(grid.data),
                 color='red', label='exact')
        plt.scatter(np.asarray(torch.flatten(grid.data)), torch.flatten(y_pred),
                    color='black', label='VPINN')

        # Set plot labels and titles
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$', rotation=0)
        plt.title('Exact and VPINN solution')

        # Show the legend
        plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)

        # Draw x-axis and grid
        plt.axhline(0, linewidth=1, linestyle='-', color='black')
        plt.grid()


    # 2d heatmap
    elif grid.dim == 2:

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(torch.reshape(u_exact(grid.data), grid_shape))
        axs[1].imshow(torch.reshape(torch.flatten(y_pred), grid_shape))

        fig.suptitle('Exact and predicted solution')

        axs[0].set_title('Exact solution')
        axs[1].set_title('VPINNS')
        for ax in axs:
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$', rotation=0)

    plt.show()


# Plots the loss over time
def plot_loss(loss_tracker: dict):

    # Plot the boundary loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_b'], label=r'boundary loss', color='red')

    # Plot the variational loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_v'], label=r'variational loss', color='blue')

    # Set labels and titles
    plt.title('Loss over time')
    plt.xlabel(r'iteration')
    plt.ylabel(r'loss', rotation=0)
    plt.yscale('log')
    plt.legend(shadow=True, loc='upper right', fontsize=18, ncol=1)
    plt.show()


# Plots the test functions
def plot_test_functions(x: Grid, n_test_func: int, *, d: int = 0):
    for i in range(1, n_test_func):
        plt.plot(x.data, dtest_function(x.data, i, d=d))
    plt.show()
