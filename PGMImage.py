import math
import numpy as np
from Image import Image


class PGMImage(Image):
    def __init__(self, rows, cols, maxLevel) -> None:
        super().__init__(rows, cols, maxLevel)
        self.type = "P2"
        self.__data = np.zeros((rows, cols), dtype=np.int64)

    def readFromFile(filepath):
        with open(filepath) as f:
            lines = f.readlines()
            image = PGMImage(0, 0, 0)
            image.type = lines[0].strip()
            i = 1
            if lines[i].startswith('#'):
                i += 1
            image.cols, image.rows = [int(x) for x in lines[i].split()]
            i += 1
            image.maxLevel = int(lines[i].strip())
            i += 1
            allPixels = []
            for line in lines[i:]:
                pixelsInLine = [int(x) for x in line.split()]
                allPixels.extend(pixelsInLine)
            if len(allPixels) != image.rows * image.cols:
                raise Exception("Image is not well formatted")
            image.__data = np.array(allPixels).reshape(
                (image.rows, image.cols))

            return image

    def writeToFile(self, filepath):
        with open(filepath, "w") as f:
            f.writelines([self.type + '\n'])
            f.writelines([f'{self.cols} {self.rows}\n'])
            f.writelines([str(self.maxLevel) + '\n'])
            f.writelines([' '.join(str(pixel) for pixel in row) + '\n'
                         for row in self.__data])

    def getHistogram(self):
        return self._histogram(self.__data)

    def getCummulatedHistogram(self):
        return self._cumulatedHistogram(self.__data)

    def getMean(self):
        return self._mean(self.__data)

    def getVariance(self):
        return self._variance(self.__data)

    def equalizeHistogram(self):
        self._equalizeHist(self.__data)
        return self

    def linearTransform(self, points):
        self._linearTransform(self.__data, points)
        return self

    def addNoise(self):
        self._addNoise(self.__data)
        return self

    def applyLinearFilter(self, filter: 'np.ndarray'):
        self._applyLinearFilter(self.__data, filter)
        return self

    def applyMedianFilter(self, n: 'int', m: 'int'):
        self._applyMedianFilter(self.__data, n, m)
        return self

    def signalToNoiseRatio(original: 'PGMImage', treated: 'PGMImage'):
        var = original.getVariance() * original.rows * original.cols
        det = 0
        for r in range(original.rows):
            for c in range(original.cols):
                det += (original.__data[r][c] - treated.__data[r][c]) ** 2
        return math.sqrt(var / det)
