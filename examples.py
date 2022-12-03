from PGMImage import PGMImage
from PPMImage import PPMImage
import helpers
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import copy


def equalizeHist():
    filepath = './images/balloons.ascii.pgm'
    balloon = PGMImage.readFromFile(filepath=filepath)

    fig, _ = helpers.plot_histogram(balloon.getHistogram())
    fig.suptitle("normal")

    balloon.equalizeHistogram()
    balloon.writeToFile(helpers.getOutputFilePath("equalized.pgm"))

    fig, _ = helpers.plot_histogram(balloon.getHistogram())
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
        noisyChat = copy.deepcopy(chat).addNoise()
        noisyChat.writeToFile(helpers.getOutputFilePath("noisy-chat.pgm"))

    filter = np.ones((3, 3)) * 1/9
    meanFiltered = copy.deepcopy(noisyChat).applyLinearFilter(filter)
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


def thresholdPPMImage():
    filepath = './images/roi.jpg'
    image = PPMImage.convertImageToPPM(filepath)

    image.rgbThreshold(50, 50, 50)
    image.writeToFile(helpers.getOutputFilePath('blackbuck.ppm'))


def otsu():
    filepath = './images/car.jpeg'
    image = PPMImage.convertImageToPPM(filepath)
    o1, o2, o3 = image.otsu()
    print(o1)
    print(o2)
    print(o3)
    image.rgbThreshold(o1, o2, o3)
    image.writeToFile(helpers.getOutputFilePath('otsu.ppm'))


def otsuCv():
    filepath = './images/roi.jpg'
    img = cv.imread(filepath)
    b, g, r = cv.split(img)
    tb, bdst = cv.threshold(b, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    tg, gdst = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    tr, rdst = cv.threshold(r, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(tb, tg, tr)
    dst = cv.merge((bdst, gdst, rdst))
    cv.imshow('image', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()


def convertImageToPPM():
    filepath = './images/roi.jpg'
    image = PPMImage.convertImageToPPM(filepath)

    image.writeToFile(helpers.getOutputFilePath("roi.ppm"))


def readWritePPM():
    filepath = './images/roi.ppm'
    image = PPMImage.readFromFile(filepath)

    image.writeToFile(helpers.getOutputFilePath("roi.ppm"))


def pgmtoppm():
    filepath = './images/chat.pgm'
    chat = PGMImage.readFromFile(filepath=filepath)
    chatppm = PPMImage(chat.rows, chat.cols, chat.maxLevel)
    chatppm._PPMImage__r = chatppm._PPMImage__g = chatppm._PPMImage__b = chat._PGMImage__data
    chatppm.writeToFile(helpers.getOutputFilePath("chatppm.ppm"))
