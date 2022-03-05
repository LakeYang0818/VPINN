from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import seaborn as sns
import torch
import yaml

from .functions import test_function, dtest_function, f, u as u_exact
from .Types.Grid import Grid

# save plots to new folder

folder_name = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
os.makedirs(os.path.join(os.path.join(os.getcwd(), 'Results'), folder_name))

"""Plots used in the VPINNS model"""

# Generate a text box string from the config
def info_from_cfg(cfg):

    dim = cfg['space']['dimension']

    pde_info = r"$\textbf{"+f"{dim}-dimensional {cfg['PDE']['type']} equation"+r"}$"
    net_info = f"Neural net: "+\
               f"{cfg['architecture']['layers']} layers, {cfg['architecture']['nodes_per_layer']} neurons per layer "

    grid_info = f"Grid: ${cfg['space']['boundary']['x']}$"
    if dim == 2:
        grid_info += fr" $\times {cfg['space']['boundary']['y']}$"
    grid_info += fr", $n_x=${cfg['space']['grid_size']['x']}"
    if dim == 2:
        grid_info += fr", $n_y=${cfg['space']['grid_size']['y']}"
    var_form_info = f"Variational form: {cfg['variational_form']}"
    test_func_info = fr"Number of test functions: $K_x=$ {cfg['N_test_functions']['x']}"
    if dim == 2:
        test_func_info += fr", $K_y=$ {cfg['N_test_functions']['y']}"
    iterations_info = fr"{cfg['N_iterations']} iterations, $\tau_b=${cfg['boundary_loss_weight']}, $\tau_v=${cfg['variational_loss_weight']}"

    return pde_info+'\n \n'+'\n'.join([net_info, grid_info, var_form_info, test_func_info, iterations_info])


# Plots the prediction value
def plot_prediction(cfg, grid: Grid, y_pred, *, grid_shape: tuple):

    # 1D plot
    if grid.dim == 1:

        fig, ax = plt.subplots()
        # Plot the exact solution and the model predictions
        ax.plot(grid.data, u_exact(grid.data),
                 color='orangered', label='exact')
        ax.scatter(np.asarray(torch.flatten(grid.data)), torch.flatten(y_pred),
                    color='black', label='VPINN')

        # Set plot labels and titles
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u$', rotation=0)
        plt.title('Exact and VPINN solution')

        # Show the legend
        plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)

        # Add info string
        try:
            info_str = info_from_cfg(cfg)
        except:
            info_str = "(Error obtaining info string; check latex settings)"

        # Add L infty norm to info box
        l_inf_err = torch.round(1000*torch.abs(torch.max(torch.subtract(u_exact(grid.data), y_pred)))).numpy()/1000
        info_str += '\n'+fr"$L^\infty$ error: {l_inf_err}"

        ax.text(0.55, 0.08, info_str, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


        # Draw x-axis and grid
        plt.axhline(0, linewidth=1, linestyle='-', color='black')
        plt.grid()
        plt.savefig('Results/'+folder_name+'/results_1d.pdf')
        plt.close()


    # 2d heatmap
    elif grid.dim == 2:

        plot_titles = ['Exact solution', 'VPINN prediction', 'Pointwise error', 'Forcing']
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        axs = np.resize(axs, (1, 4))[0]

        # Generate the datasets
        exact_solution = torch.reshape(u_exact(grid.data), grid_shape)
        predicted_solution = torch.reshape(torch.flatten(y_pred), grid_shape)
        err = np.abs(predicted_solution-exact_solution)

        # Plot the heatmaps
        extent = (grid.x[0].numpy()[0], grid.x[-1].numpy()[0], grid.y[0].numpy()[0], grid.y[-1].numpy()[0])
        im1 = axs[0].imshow(exact_solution, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True))
        im2 = axs[1].imshow(predicted_solution, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True))
        im3 = axs[2].imshow(err, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True))
        imgs = [im1, im2, im3]

        # Add colorbars to all the heatmaps
        for i in range(len(imgs)):
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            fig.colorbar(imgs[i], cax=cax, orientation='vertical')
            axs[i].set_title(plot_titles[i])
            if i > 1:
                axs[i].set_xlabel(r'$x$')
            if i == 0 or i == 2:
                axs[i].set_ylabel(r'$y$', rotation=0)
                axs[i].yaxis.labelpad = 10

        # Write the text box
        axs[3].axis('off')
        try:
            info_str = info_from_cfg(cfg)
        except:
            info_str = "(Error obtaining info string; check latex settings)"

        l_inf_err = torch.round(1000*torch.abs(torch.max(err))).numpy()/1000
        info_str += '\n'+fr"$L^\infty$ error: {l_inf_err}"

        axs[3].text(0.15, 1.0, info_str, transform=axs[3].transAxes,
                    verticalalignment='top')

        # Save the file
        fig.savefig('Results/'+folder_name+'/results_2d.pdf')
        plt.close()


# Plots the loss over time
def plot_loss(loss_tracker: dict):

    # Plot the boundary loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_b'], label=r'boundary loss', color='darkred')

    # Plot the variational loss
    plt.plot(loss_tracker['iter'], loss_tracker['loss_v'], label=r'variational loss', color='navy')

    plt.grid()

    # Set labels and titles
    plt.title('Loss over time')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'Total loss', rotation=90)
    plt.yscale('log')
    plt.legend(shadow=True, loc='upper right', fontsize=18, ncol=1)
    plt.savefig('Results/' + folder_name + '/loss.pdf')


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

        plt.savefig('Results/' + folder_name + '/test_functions_1d.pdf')

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

        plt.savefig('Results/' + folder_name + '/test_functions_2d.pdf')

# Writes the config
def write_config(cfg):

    with open(f'Results/' + folder_name + '/config.yaml', 'w') as file:
        yaml.dump(cfg, file)



