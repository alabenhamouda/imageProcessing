import numpy as np
import pathlib
import cv2
from Image import Image
from PGMImage import PGMImage


class PPMImage(Image):
    def __init__(self, rows, cols, maxLevel) -> None:
        super().__init__(rows, cols, maxLevel)
        self.type = "P3"
        self.__r = np.zeros((rows, cols), dtype=np.int64)
        self.__g = np.zeros((rows, cols), dtype=np.int64)
        self.__b = np.zeros((rows, cols), dtype=np.int64)

    def __getitem__(self, pos):
        last = None
        if isinstance(pos, tuple):
            if len(pos) > 3:
                raise Exception("Invalid length of slice objects tuple")
            if len(pos) == 3:
                pos, last = pos[:-1], pos[-1]

        red = self.__r[pos]
        green = self.__g[pos]
        blue = self.__b[pos]
        ret = np.stack([red, green, blue], axis=np.ndim(red))
        if last != None:
            if np.ndim(ret) == 1:
                ret = ret[last]
            else:
                ret = ret[:, :, last]
        return ret

    def readFromFile(filepath):
        with open(filepath) as f:
            lines = f.readlines()
            image = PPMImage(0, 0, 0)
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
            if len(allPixels) != image.rows * image.cols * 3:
                raise Exception("Image is not well formatted")
            data = np.array(allPixels)
            data = data.reshape((image.rows, image.cols, 3))
            image.__r = data[:, :, 0]
            image.__g = data[:, :, 1]
            image.__b = data[:, :, 2]

            return image

    def writeToFile(self, filepath):
        with open(filepath, "w") as f:
            f.writelines([self.type + '\n'])
            f.writelines([f'{self.cols} {self.rows}\n'])
            f.writelines([str(self.maxLevel) + '\n'])
            data = self[:, :]
            datastr = '\n'.join([' '.join([' '.join(str(value)
                                                    for value in pixel) for pixel in row]) for row in data])
            f.writelines([datastr])

    def convertImageToPPM(filepath: 'str') -> 'PPMImage':
        extension = pathlib.Path(filepath).suffix
        if extension == ".ppm":
            return PPMImage.readFromFile(filepath)
        elif extension == ".pgm":
            pgmImage = PGMImage.readFromFile(filepath)
            data = pgmImage._PGMImage__data
            ppmImage = PPMImage(
                pgmImage.rows, pgmImage.cols, pgmImage.maxLevel)
            ppmImage.__r = data
            ppmImage.__g = np.array(data)
            ppmImage.__b = np.array(data)
            return ppmImage
        else:
            img = cv2.imread(filepath)
            rows, cols, _ = img.shape
            maxLevel = img.max()
            ppmImage = PPMImage(rows, cols, maxLevel)
            ppmImage.__r = img[:, :, 0]
            ppmImage.__g = img[:, :, 1]
            ppmImage.__b = img[:, :, 2]

            return ppmImage

    def rgbThreshold(self, tr, tg, tb):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.__r[r][c] < tr:
                    self.__r[r][c] = 0
                else:
                    self.__r[r][c] = self.maxLevel
                if self.__g[r][c] < tg:
                    self.__g[r][c] = 0
                else:
                    self.__g[r][c] = self.maxLevel
                if self.__b[r][c] < tb:
                    self.__b[r][c] = 0
                else:
                    self.__b[r][c] = self.maxLevel
        return self

    def andThreshold(self, threshold):
        for r in range(self.rows):
            for c in range(self.cols):
                pixel = self[r, c]
                if pixel[0] < threshold or pixel[1] < threshold or pixel[2] < threshold:
                    self.__r[r][c] = self.__g[r][c] = self.__b[r][c] = 0

        return self

    def orThreshold(self, threshold):
        for r in range(self.rows):
            for c in range(self.cols):
                pixel = self[r, c]
                if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
                    self.__r[r][c] = self.__g[r][c] = self.__b[r][c] = 0

        return self

    def probability(self, color: 'int'):
        ret = np.zeros(self.maxLevel + 1)
        for r in range(self.rows):
            for c in range(self.cols):
                ret[self[r, c, color]] += 1

        ret /= self.rows * self.cols
        return ret

    def cumulatedProbability(self, color: 'int'):
        ret = self.probability(color)
        ret = np.cumsum(ret)
        return ret

    def __otsu(self, color):
        prob = self.probability(color)
        probCumul = np.cumsum(prob)
        a = np.arange(self.maxLevel + 1)
        mean = a * prob
        mean = np.cumsum(mean)

        val = np.inf
        thresh = -1
        for lvl in range(0, self.maxLevel):
            q1 = probCumul[lvl]
            q2 = probCumul[self.maxLevel] - q1
            m1 = mean[lvl] / q1
            m2 = (mean[self.maxLevel] - mean[lvl]) / q2
            p1, p2 = np.hsplit(prob, [lvl + 1])
            a1, a2 = np.hsplit(a, [lvl + 1])
            v1 = np.sum(((a1 - m1) ** 2) * p1) / q1
            v2 = np.sum(((a2 - m2) ** 2) * p2) / q2
            v = q1 * v1 + q2 * v2
            if v < val:
                val = v
                thresh = lvl

        return thresh

    def otsu(self):
        return self.__otsu(0), self.__otsu(1), self.__otsu(2)

    def applyLinearFilter(self, filter: 'np.ndarray'):
        self._applyLinearFilter(self.__r, filter)
        self._applyLinearFilter(self.__g, filter)
        self._applyLinearFilter(self.__b, filter)
        return self

    def applyMedianFilter(self, n, m):
        self._applyMedianFilter(self.__r, n, m)
        self._applyMedianFilter(self.__g, n, m)
        self._applyMedianFilter(self.__b, n, m)
        return self
