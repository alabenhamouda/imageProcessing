from Image import PGMImage
import helpers
import os
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

    noisyChatPath = helpers.getOutputFilePath("noisy-chat.pgm")
    if os.path.exists(noisyChatPath):
        noisyChat = PGMImage.readFromFile(noisyChatPath)
    else:
        noisyChat = chat.addNoise()
        noisyChat.writeToFile(helpers.getOutputFilePath("noisy-chat.pgm"))

    filter = np.ones((3, 3)) * 1/9
    filtered = noisyChat.applyLinearFilter(filter)
    filtered.writeToFile(helpers.getOutputFilePath("mean-filtered.pgm"))


def medianFilter():
    filepath = './images/chat.pgm'
    chat = PGMImage.readFromFile(filepath=filepath)

    noisyChatPath = helpers.getOutputFilePath("noisy-chat.pgm")
    if os.path.exists(noisyChatPath):
        noisyChat = PGMImage.readFromFile(noisyChatPath)
    else:
        noisyChat = chat.addNoise()
        noisyChat.writeToFile(helpers.getOutputFilePath("noisy-chat.pgm"))

    filtered = noisyChat.applyMedianFilter(3, 3)
    filtered.writeToFile(helpers.getOutputFilePath("median-filtered.pgm"))


def enhanceImage():
    filepath = './images/chat.pgm'
    chat = PGMImage.readFromFile(filepath=filepath)

    filter = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    enhanced = chat.applyLinearFilter(filter)
    enhanced.writeToFile(helpers.getOutputFilePath("enhanced-chat.pgm"))


def compareMedianAndMean():
    filepath = './images/chat.pgm'
    chat = PGMImage.readFromFile(filepath=filepath)

    noisyChatPath = helpers.getOutputFilePath("noisy-chat.pgm")
    if os.path.exists(noisyChatPath):
        noisyChat = PGMImage.readFromFile(noisyChatPath)
    else:
        noisyChat = chat.addNoise()
        noisyChat.writeToFile(helpers.getOutputFilePath("noisy-chat.pgm"))

    filter = np.ones((3, 3)) * 1/9
    meanFiltered = noisyChat.applyLinearFilter(filter)
    meanFiltered.writeToFile(helpers.getOutputFilePath("mean-filtered.pgm"))

    medianFiltered = noisyChat.applyMedianFilter(3, 3)
    medianFiltered.writeToFile(
        helpers.getOutputFilePath("median-filtered.pgm"))

    meanSNR = PGMImage.signalToNoiseRatio(chat, meanFiltered)
    medianSNR = PGMImage.signalToNoiseRatio(chat, medianFiltered)

    print(f"Signal To Noise Ratio of the mean filtered image {meanSNR}")
    print(f"Signal To Noise Ratio of the median filtered image {medianSNR}")


def edgeDetection():
    filepath = './images/chat.pgm'
    chat = PGMImage.readFromFile(filepath=filepath)

    filter1 = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    filter2 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])

    result = chat.applyLinearFilter(filter1).applyLinearFilter(filter2)
    result.writeToFile(helpers.getOutputFilePath("edge-detection.pgm"))
