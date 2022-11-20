from Image import PGMImage
import matplotlib.pyplot as plt

filepath = './images/balloons.ascii.pgm'
balloon = PGMImage.readFromFile(filepath=filepath)

fig, ax = plt.subplots()
levels = [i for i in range(balloon.maxLevel + 1)]
histogram = balloon.getHistogram()
ax.bar(levels, histogram)
plt.show()
