import matplotlib.pyplot as plt


def plot_histogram(hist):
    levels = [i for i in range(len(hist))]
    fig, ax = plt.subplots()
    ax.bar(levels, hist)
