from Image import PGMImage
import helpers
import numpy as np
import matplotlib.pyplot as plt


def equalizeHist():
    filepath = './images/balloons.ascii.pgm'
    balloon = PGMImage.readFromFile(filepath=filepath)

    fig, _ = helpers.plot_histogram(balloon.getHistogram())
    fig.suptitle("normal")

    equalizedBalloon = balloon.getEqualizedHistImage()
    equalizedBalloon.writeToFile(helpers.getOutputFilePath("equalized.pgm"))

    fig, _ = helpers.plot_histogram(equalizedBalloon.getHistogram())
    fig.suptitle("equalized")

    plt.show()


def linearTransform():
    filepath = './images/balloons.ascii.pgm'
    balloon = PGMImage.readFromFile(filepath=filepath)

    fig, _ = helpers.plot_histogram(balloon.getHistogram())
    fig.suptitle("normal")

    transformationPoints = ((100, 200), (200, 100))
    transformed = balloon.linearTransform(transformationPoints)
    transformed.writeToFile(
        helpers.getOutputFilePath("linearTransformation.pgm"))
    fig, _ = helpers.plot_histogram(transformed.getHistogram())
    fig.suptitle("transformed")

    plt.show()


def linearFilter():
    filepath = './images/chat.pgm'
    chat = PGMImage.readFromFile(filepath=filepath)

    noisyChat = chat.addNoise()
    noisyChat.writeToFile(helpers.getOutputFilePath("noisy-chat.pgm"))

    filter = np.ones((3, 3)) * 1/9
    filtered = noisyChat.applyFilter(filter)
    filtered.writeToFile(helpers.getOutputFilePath("mean-filtered.pgm"))