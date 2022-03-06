import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import torch
import sys
sys.path.append("..")

from function_definitions import u as u_exact
from .tools import output_dir, info_from_cfg
from ..Types import Grid


"""Plots the model prediction and compares it to the exact solution, if given"""


def plot_prediction(cfg, grid: Grid, y_pred, *, grid_shape: tuple, show: bool = True):
    # For the Burger's equation, plot four time snaps
    if cfg['PDE']['type'] == 'Burger':
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        axs = np.resize(axs, (1, 4))[0]

        y_pred = torch.reshape(y_pred, (len(grid.x), len(grid.y)))

        for i in range(len(axs)):
            t = int((len(y_pred) - 1) / 3 * i)
            axs[i].scatter(grid.x, y_pred[t],
                           color='black', label='VPINN')
            if i > 1:
                axs[i].set_xlabel(r'$x$')
            if i == 0 or i == 2:
                axs[i].set_ylabel(r'$y$', rotation=0)
                axs[i].yaxis.labelpad = 10
            axs[i].text(0.05, 0.9, fr'$t={np.around(grid.y[-1].numpy()[0] * (t + 1) / grid_shape[-1], 3)}$',
                        transform=axs[i].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if show:
            plt.show()
        else:
            fig.savefig('Results/' + output_dir + '/Burger_snapshots.pdf')
            plt.close()

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
        l_inf_err = np.around(torch.abs(torch.max(torch.subtract(u_exact(grid.data), y_pred))).numpy())
        info_str += '\n' + fr"$L^\infty$ error: {l_inf_err}"

        ax.text(0.55, 0.08, info_str, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Draw x-axis and grid
        plt.axhline(0, linewidth=1, linestyle='-', color='black')
        plt.grid()
        if show:
            plt.show()
        else:
            plt.savefig('Results/' + output_dir + '/results_1d.pdf')
            plt.close()

    # 2d heatmap
    elif grid.dim == 2:
        if cfg['PDE']['type'] != 'Burger':
            plot_titles = ['Exact solution', 'VPINN prediction', 'Pointwise error', 'Forcing']
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            axs = np.resize(axs, (1, 4))[0]

            # Generate the datasets
            exact_solution = torch.reshape(u_exact(grid.data), grid_shape)
            predicted_solution = torch.reshape(torch.flatten(y_pred), grid_shape)
            err = np.abs(predicted_solution - exact_solution)

            # Plot the heatmaps
            extent = (grid.x[0].numpy()[0], grid.x[-1].numpy()[0], grid.y[0].numpy()[0], grid.y[-1].numpy()[0])
            im1 = axs[0].imshow(exact_solution, origin='lower', extent=extent,
                                cmap=sns.color_palette("rocket", as_cmap=True), aspect='auto')
            im2 = axs[1].imshow(predicted_solution, origin='lower', extent=extent,
                                cmap=sns.color_palette("rocket", as_cmap=True), aspect='auto')
            im3 = axs[2].imshow(err, origin='lower', extent=extent, cmap=sns.color_palette("rocket", as_cmap=True),
                                aspect='auto')
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

            l_inf_err = torch.round(1000 * torch.abs(torch.max(err))).numpy() / 1000
            info_str += '\n' + fr"$L^\infty$ error: {l_inf_err}"

            axs[3].text(0.0, 1.0, info_str, transform=axs[3].transAxes,
                        verticalalignment='top')

        else:
            fig, axs = plt.subplots(1, 2)
            axs = np.resize(axs, (1, 2))[0]

            # Generate the datasets
            predicted_solution = torch.reshape(torch.flatten(y_pred), grid_shape)

            # Plot the heatmaps
            extent = (grid.x[0].numpy()[0], grid.x[-1].numpy()[0], grid.y[0].numpy()[0], grid.y[-1].numpy()[0])
            img = axs[0].imshow(predicted_solution, origin='lower', extent=extent,
                                cmap=sns.color_palette("rocket", as_cmap=True), aspect='auto')

            # Add colorbars to all the heatmaps
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            fig.colorbar(img, cax=cax, orientation='vertical')
            axs[0].set_xlabel(r'$x$')
            axs[0].set_ylabel(r'$y$', rotation=0)
            axs[0].yaxis.labelpad = 10

            # Write the text box
            axs[1].axis('off')
            try:
                info_str = info_from_cfg(cfg)
            except:
                info_str = "(Error obtaining info string; check latex settings)"

            axs[1].text(0.0, 1.0, info_str, transform=axs[1].transAxes,
                        verticalalignment='top')

        # Save the file
        if show:
            plt.show()
        else:
            fig.savefig('Results/' + output_dir + '/prediction.pdf')
            plt.close()
