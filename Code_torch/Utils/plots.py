import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import torch

from .functions import test_function, dtest_function, f, u as u_exact
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
        plt.savefig('Results/results_1d.pdf')
        plt.close()


    # 2d heatmap
    elif grid.dim == 2:

        plot_titles = ['Exact solution', 'VPINN prediction', 'Pointwise error', 'Forcing']
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        axs = np.resize(axs, (1, 4))[0]

        exact_solution = torch.reshape(u_exact(grid.data), grid_shape)
        predicted_solution = torch.reshape(torch.flatten(y_pred), grid_shape)
        err = np.abs(predicted_solution-exact_solution)
        forcing = torch.reshape(f(grid.data), grid_shape)

        extent = (grid.x[0].numpy()[0], grid.x[-1].numpy()[0], grid.y[0].numpy()[0], grid.y[-1].numpy()[0])
        im1 = axs[0].imshow(exact_solution, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True))
        im2 = axs[1].imshow(predicted_solution, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True))
        im3 = axs[2].imshow(err, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True))
        im4 = axs[3].imshow(forcing, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True))

        imgs = [im1, im2, im3, im4]

        for i in range(len(imgs)):
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            fig.colorbar(imgs[i], cax=cax, orientation='vertical')
            axs[i].set_title(plot_titles[i])
            if i > 2:
                axs[i].set_xlabel(r'$x$')
            if i == 0 or i == 3:
                axs[i].set_ylabel(r'$y$', rotation=0)
                axs[i].yaxis.labelpad = 10

        fig.savefig('Results/results_2d.pdf')
        plt.close()


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


# Plots the test functions and their derivatives up to order n
def plot_test_functions(grid: Grid, *, order: int, d: int = 1):

    fig, axs = plt.subplots(int(order / 2), 2, sharex=True)
    axs = np.resize(axs, (1, order))[0]

    if grid.dim == 1:

        fig.suptitle(fr'First {order} test functions and derivatives')

        for i in range(order):
            axs[i].axhline(0, linewidth=1, linestyle='-', color='black')
            axs[i].plot(grid.data, test_function(grid.data, i+1), color='peru', label=fr'$v_{str(i)}(x)$')
            if d > 0:
                label_derivs = fr'$v^\prime_{str(i)}(x)$' if d==1 else fr'$d^{str(d)}v_{str(i)}(x)$'
                axs[i].plot(grid.data, dtest_function(grid.data, i+1, d=d), color='darkred', label=label_derivs)
            axs[i].legend(ncol=2)

        plt.show()
        fig.savefig('Results/test_functions_1d.pdf')

    elif grid.dim == 2:

        if d > 0:
            fig.suptitle(fr'Derivatives of order {d} of test functions')
        else:
            fig.suptitle(fr'First {order} test functions')

        extent = (grid.x[0].numpy()[0], grid.x[-1].numpy()[0], grid.y[0].numpy()[0], grid.y[-1].numpy()[0])
        for i in range(order):
            img = axs[i].imshow(torch.reshape(torch.prod(dtest_function(grid.data, i+1, d=d), dim=1), (len(grid.x), len(grid.y))),
                       cmap=sns.color_palette("mako", as_cmap=True), origin='lower', extent=extent)
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            fig.colorbar(img, cax=cax, orientation='vertical')

        plt.show()
        fig.savefig('Results/dtest_functions_2d.pdf')


