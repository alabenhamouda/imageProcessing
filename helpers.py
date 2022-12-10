import matplotlib.pyplot as plt
import os
import numpy as np


def plot_histogram(hist):
    levels = [i for i in range(len(hist))]
    fig, ax = plt.subplots()

    ax.fill_between(levels, hist, step="pre", alpha=0.4)
    ax.plot(levels, hist, drawstyle="steps")

    return fig, ax

def plot_points(ax: plt.Axes, points: np.ndarray):
    x = points[:,0]
    y = points[:,1]

    ax.clear()
    ax.plot(x, y)


def getOutputFilePath(filename):
    if not os.path.exists("./out"):
        os.mkdir("./out")
    return f"./out/{filename}"
