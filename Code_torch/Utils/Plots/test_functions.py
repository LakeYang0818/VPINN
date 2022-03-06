import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import torch

from ..test_functions import dtest_function, test_function
from ..utils import rescale_grid
from .tools import output_dir
from ..Types import Grid


"""Plots the test functions and their derivatives up to order n"""


def plot_test_functions(grid: Grid, *, order: int, d: int = 1, show: bool = False):

    fig, axs = plt.subplots(int(order / 2), 2, sharex=True)
    axs = np.resize(axs, (1, order))[0]

    grid = rescale_grid(grid)

    if grid.dim == 1:

        fig.suptitle(fr'First {order} test functions and derivatives')

        for i in range(order):
            axs[i].axhline(0, linewidth=1, linestyle='-', color='black')
            axs[i].plot(grid.data, test_function(grid.data, i+1), color='peru', label=fr'$v_{str(i)}(x)$')
            if d > 0:
                label_derivs = fr'$v^\prime_{str(i)}(x)$' if d==1 else fr'$d^{str(d)}v_{str(i)}(x)$'
                axs[i].plot(grid.data, dtest_function(grid.data, i+1, d=d), color='darkred', label=label_derivs)
            axs[i].legend(ncol=2)

        plt.savefig('Results/' + output_dir + '/test_functions_1d.pdf')

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

        plt.savefig('Results/' + output_dir + '/test_functions_2d.pdf')

    if show:
        plt.show()
    else:
        plt.close()