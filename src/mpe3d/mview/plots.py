"""Matplotlib helpers used by the MDS/TSNE/MPSE classes for quick diagnostics."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np


def plot_cost(cost, steps=None, title='computations', plot=True, ax=None):
    """Plot a cost (and optional step-size) curve on a log scale."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        plot = False
    if steps is not None:
        ax.semilogy(steps, label='step size', linestyle='--')
    ax.semilogy(cost, label='cost', linewidth=3)
    ax.set_xlabel('iterations')
    ax.legend()
    ax.set_title(title)
    if plot is True:
        plt.draw()
        plt.pause(1.0)


def plot2D(Y, colors=None, edges=None, labels=None, title=None, axis=True,
           ax=None, plot=True, markersize=40, weight=None, **kwargs):
    """Scatter plot of a 2D image (one projected perspective)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        plot = False

    if edges is not None:
        for i, j in edges:
            ax.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], '-',
                    linewidth=0.2, color='gray', zorder=1)

    ax.scatter(Y[:, 0], Y[:, 1], s=markersize, c=colors, zorder=2)
    if weight is not None:
        # Points hidden in this perspective (zero weight) are drawn black.
        for index in range(len(Y) // 2):
            if weight[index * (len(Y) - 1)] == 0:
                ax.scatter(Y[index][0], Y[index][1], s=markersize, c='black',
                           zorder=2)

    if labels is not None:
        N = len(Y)
        if labels is True:
            labels = range(N)
        for i in range(N):
            ax.annotate(labels[i], (Y[i, 0], Y[i, 1]), textcoords="offset points",
                        xytext=(0, 4), ha='center', fontsize='large',
                        fontstyle='oblique', fontweight='bold',
                        horizontalalignment='center', color='red')
    plt.title(title, fontweight='bold', fontsize='large')

    if axis is False:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    if plot is True:
        plt.draw()
        plt.pause(1)


def plot3D(X, perspectives=None, edges=None, colors=None, title=None,
           axis=False, ax=None, markersize=40, labels=None):
    """Scatter plot of a 3D embedding, optionally showing perspective rays."""
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        plot = True
    else:
        fig = ax.figure
        plot = False

    if perspectives is not None:
        q = perspectives
        for k in range(len(q)):
            ind = np.argmax(np.sum(q[k] * X, axis=1))
            m = np.linalg.norm(X[ind]) / np.linalg.norm(q[k])
            ax.plot([0, m * q[k][0]], [0, m * q[k][1]], [0, m * q[k][2]], '--',
                    linewidth=4.5, color='gray')
    if edges is not None:
        for i, j in edges:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]],
                    '-', linewidth=0.1, color='lightgray')

    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=colors, s=markersize)

    if labels is not None:
        N = len(X)
        if labels is True:
            labels = range(N)
        for i in range(N):
            ax.text(X[i, 0], X[i, 1], X[i, 2], labels[i], fontsize='large',
                    fontstyle='oblique', fontweight='bold',
                    horizontalalignment='center', color='red')
    if axis is False:
        ax.set_axis_off()
    if title is not None:
        ax.title.set_text(title)
    ax.grid(color='r')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.setp(ax.spines.values(), color='blue')
    ax.axes.get_xaxis().set_visible(False)

    if plot is True:
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    return fig
