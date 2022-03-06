import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from .tools import output_dir
from ..Types import Grid


"""Animated plot of equation solution over time"""


def animate(grid: Grid, y_pred, *, n_frames: int=200, fps: int=30, interval: int=200):
    fig = plt.figure()
    ax = plt.axes(xlim=(grid.x[0], grid.x[-1]), ylim=(grid.y[0], grid.y[-1]))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')

    y_pred = torch.reshape(y_pred, (len(grid.x), len(grid.y)))

    line, = ax.plot([], [], color='black', lw='3')

    def init():
        line.set_data(grid.x, y_pred[0])
        return line,

    def anim(i):
        t = int((len(y_pred) - 1) / n_frames * i)
        line.set_data(grid.x, y_pred[t])
        return line,

    anim = animation.FuncAnimation(fig, anim, init_func=init, frames=n_frames, interval=interval, blit=True)
    anim.save(f'Results/' + output_dir + '/animation.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()
