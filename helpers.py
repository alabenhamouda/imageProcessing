import matplotlib.pyplot as plt
import os


def plot_histogram(hist):
    levels = [i for i in range(len(hist))]
    fig, ax = plt.subplots()

    ax.fill_between(levels, hist, step="pre", alpha=0.4)
    ax.plot(levels, hist, drawstyle="steps")

    return fig, ax


def getOutputFilePath(filename):
    if not os.path.exists("./out"):
        os.mkdir("./out")
    return f"./out/{filename}"
