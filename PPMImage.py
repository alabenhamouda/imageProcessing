import numpy as np
import copy
import math


class PPMImage:
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

    def __meanPerClass(self, color: 'int'):
        prob = self.probability(color)
        for i in range(self.maxLevel + 1):
            prob[i] *= i
        return np.cumsum(prob)

    def __classProperty(cumulatedList: 'list', sep: 'int', lower: 'bool'):
        if lower:
            if sep == 0:
                return 0
            else:
                return cumulatedList[sep - 1]
        else:
            return cumulatedList[len(cumulatedList) - 1] - \
                PPMImage.__classProperty(
                    cumulatedList, sep, lower=True)

    def __otsu(self, color):
        probCumul = self.cumulatedProbability(color)
        mean = self.__meanPerClass(color)
        var = float('-inf')
        for sep in range(1, self.maxLevel + 1):
            w0 = PPMImage.__classProperty(probCumul, sep, lower=True)
            w1 = PPMImage.__classProperty(probCumul, sep, lower=False)
            mean0 = PPMImage.__classProperty(mean, sep, lower=True)
            mean0 /= w0
            mean1 = PPMImage.__classProperty(mean, sep, lower=False)
            mean1 /= w1

            newVar = w0 * w1 * ((mean0 - mean1) ** 2)
            if newVar > var:
                ret = sep
                newVar = var

        return ret

    def otsu(self):
        return self.__otsu(0), self.__otsu(1), self.__otsu(2)
