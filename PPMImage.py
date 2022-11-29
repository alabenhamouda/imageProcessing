import numpy as np
import copy
import cv2


class PPMImage:
    def __init__(self) -> None:
        self.type = "P3"
        self.rows = self.cols = 0
        self.maxLevel = 0
        self.data = None

    def readFromFile(filepath):
        with open(filepath) as f:
            lines = f.readlines()
            image = PPMImage()
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
            image.data = []
            for i in range(0, image.rows * image.cols * 3, image.cols * 3):
                rowPixels = allPixels[i:i+image.cols * 3]
                row = []
                for j in range(0, image.cols * 3, 3):
                    row.append(rowPixels[j:j + 3])
                image.data.append(row)

            return image

    def writeToFile(self, filepath):
        with open(filepath, "w") as f:
            f.writelines([self.type + '\n'])
            f.writelines([f'{self.cols} {self.rows}\n'])
            f.writelines([str(self.maxLevel) + '\n'])
            datastr = '\n'.join([' '.join([' '.join(str(value)
                                                    for value in pixel) for pixel in row]) for row in self.data])
            f.writelines([datastr])

    def convertImageToPPM(filepath: 'str') -> 'PPMImage':
        img = cv2.imread(filepath)
        ppmImage = PPMImage()
        ppmImage.rows, ppmImage.cols, _ = img.shape
        ppmImage.data = img[:, :]
        ppmImage.maxLevel = img.max()

        return ppmImage

    def rgbThreshold(self, tr, tg, tb):
        image = copy.deepcopy(self)
        for r in range(image.rows):
            for c in range(image.cols):
                if image.data[r][c][0] < tr:
                    image.data[r][c][0] = 0
                else:
                    image.data[r][c][0] = image.maxLevel
                if image.data[r][c][1] < tg:
                    image.data[r][c][1] = 0
                else:
                    image.data[r][c][1] = image.maxLevel
                if image.data[r][c][2] < tb:
                    image.data[r][c][2] = 0
                else:
                    image.data[r][c][2] = image.maxLevel
        return image

    def andThreshold(self, threshold):
        image = copy.deepcopy(self)
        for r in range(image.rows):
            for c in range(image.cols):
                pixel = image.data[r][c]
                if pixel[0] < threshold or pixel[1] < threshold or pixel[2] < threshold:
                    image.data[r][c][0] = image.data[r][c][1] = image.data[r][c][2] = 0

        return image

    def orThreshold(self, threshold):
        image = copy.deepcopy(self)
        for r in range(image.rows):
            for c in range(image.cols):
                pixel = image.data[r][c]
                if pixel[0] < threshold and pixel[1] < threshold and pixel[2] < threshold:
                    image.data[r][c][0] = image.data[r][c][1] = image.data[r][c][2] = 0

        return image

    def probability(self, color: 'int'):
        ret = np.zeros(self.maxLevel + 1)
        for r in range(self.rows):
            for c in range(self.cols):
                ret[self.data[r][c][color]] += 1

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
