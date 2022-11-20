from Image import PGMImage
import helpers
import matplotlib.pyplot as plt

filepath = './images/balloons.ascii.pgm'
balloon = PGMImage.readFromFile(filepath=filepath)

# helpers.plot_histogram(balloon.getHistogram())
# plt.show()

print(balloon.getMean())
