import matplotlib.pyplot as plt
from .test_functions import dtest_function

from .functions import u as u_exact
from .data_types import DataGrid, Grid


# Plots the prediction value
def plot_prediction(x: Grid, y_pred, grid: Grid):
    for xc in grid.data:
        plt.axvline(x=xc, linewidth=1, ls='--')
    plt.plot(x.data, [u_exact(pt) for pt in x.data], color='red', label='exact')
    plt.scatter(x.data, y_pred, color='black', label='VPINN')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u$', rotation=0)
    plt.title('Exact and VPINN solution')
    plt.axhline(0, linewidth=0.8, linestyle='-', color='black')
    plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)
    plt.show()


# Plots the quadrature data
def plot_quadrature_data(quadrature_data: DataGrid):
    plt.title('Quadrature points')
    plt.scatter(quadrature_data.grid.data, quadrature_data.data)
    plt.show()


# Plots the loss over time
def plot_loss(loss_tracker: dict):
    plt.title('Loss over time')
    # plt.plot(loss_tracker['iter'], loss_tracker['total_loss'], label=r'total loss', color='black')
    plt.plot(loss_tracker['iter'], loss_tracker['loss_b'], label=r'boundary loss', color='red')
    plt.plot(loss_tracker['iter'], loss_tracker['loss_v'], label=r'variational loss', color='blue')
    plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)
    plt.xlabel(r'iteration')
    plt.ylabel(r'loss', rotation=0)
    plt.show()


# Plots the test functions
def plot_test_functions(x: Grid, n_test_func: int, *, d: int = 0):
    for i in range(1, n_test_func):
        plt.plot(x.data, dtest_function(x.data, i, d=d))
    plt.show()
