import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from function_definitions import u as u_exact
from .tools import output_dir
from ..Datatypes import Grid


"""Animated plot of equation solution over time"""


def animate(grid: Grid, y_pred, *, n_frames: int=200, fps: int=30, interval: int=200, show: bool = False):

    fig = plt.figure()
    y_pred = torch.reshape(y_pred, (len(grid.x), len(grid.y)))
    ax = plt.axes(xlim=(grid.x[0], grid.x[-1]), ylim=(0.9*torch.min(y_pred), 1.1*torch.max(y_pred)))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')

    y_pred = torch.reshape(y_pred, (len(grid.x), len(grid.y)))

    # grid_data = torch.reshape(grid.data, (len(grid.x), len(grid.y), 2))
    line, = ax.plot([], [], color='black', lw='3', label='VPINN')
    # exact, = ax.plot([], [], color='darkred', lw='3', label='exact')

    def init():
        line.set_data(grid.x, y_pred[0])
        # exact.set_data(grid.x, u_exact(grid_data[0]))
        return line,

    def anim(i):
        t = int((len(y_pred) - 1) / n_frames * i)
        line.set_data(grid.x, y_pred[t])
        # exact.set_data(grid.x, u_exact(grid_data[t]))
        return line,

    anim = animation.FuncAnimation(fig, anim, init_func=init, frames=n_frames, interval=interval, blit=True)
    anim.save(f'Results/' + output_dir + '/animation.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    if show:
        plt.show()
    else:
        plt.close()
