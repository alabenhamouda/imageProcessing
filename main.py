from Image import PGMImage
import helpers
import matplotlib.pyplot as plt

filepath = './images/balloons.ascii.pgm'
balloon = PGMImage.readFromFile(filepath=filepath)

fig, _ = helpers.plot_histogram(balloon.getHistogram())
fig.suptitle("normal")

equalizedBalloon = balloon.getEqualizedHistImage()
equalizedBalloon.writeToFile(helpers.getOutputFilePath("equalized.pgm"))

fig, _ = helpers.plot_histogram(equalizedBalloon.getHistogram())
fig.suptitle("equalized")

plt.show()
