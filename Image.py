import copy
import math
import random
import numpy as np
from operator import itemgetter
from linearTransformation import LinearTransformation


class PGMImage:

    def readFromFile(filepath):
        with open(filepath) as f:
            lines = f.readlines()
            image = PGMImage()
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
            image.data = []
            for i in range(0, image.rows * image.cols, image.cols):
                image.data.append(allPixels[i:i+image.cols])

            return image

    def writeToFile(self, filepath):
        with open(filepath, "w") as f:
            f.writelines([self.type + '\n'])
            f.writelines([f'{self.cols} {self.rows}\n'])
            f.writelines([str(self.maxLevel) + '\n'])
            f.writelines([' '.join(str(pixel) for pixel in row) + '\n'
                         for row in self.data])

    def getHistogram(self):
        occ = {}
        for row in self.data:
            for pixel in row:
                if pixel not in occ:
                    occ[pixel] = 0
                occ[pixel] += 1
        return [occ.get(level, 0) for level in range(self.maxLevel + 1)]

    def getCummulatedHistogram(self):
        hist = self.getHistogram()
        histCummul = [0] * len(hist)
        histCummul[0] = hist[0]
        for i in range(1, len(hist)):
            histCummul[i] = histCummul[i - 1] + hist[i]
        return histCummul

    def getMean(self):
        sum = 0
        for row in self.data:
            for pixel in row:
                sum += pixel

        return sum / (self.rows * self.cols)

    def getEqualizedHistImage(self):
        image = copy.deepcopy(self)
        histCummul = image.getCummulatedHistogram()
        lvls = [0] * (image.maxLevel + 1)
        n = image.rows * image.cols
        for i in range(image.maxLevel + 1):
            lvls[i] = math.floor(image.maxLevel / n * histCummul[i])

        for r in range(image.rows):
            for c in range(image.cols):
                image.data[r][c] = lvls[image.data[r][c]]

        return image

    def linearTransform(self, points):
        image = copy.deepcopy(self)
        points = sorted(points, key=itemgetter(0, 1))
        points.insert(0, (0, 0))
        points.append((self.maxLevel, self.maxLevel))
        lines = []
        for i in range(1, len(points)):
            if points[i - 1][0] == points[i][0]:
                raise Exception(
                    "two points cannot be on the same vertical line")
            lines.append(LinearTransformation(points[i - 1], points[i]))
        lvls = [0] * (image.maxLevel + 1)
        lineIdx = 0
        for lvl in range(image.maxLevel + 1):
            if lvl > points[lineIdx + 1][0]:
                lineIdx += 1
            lvls[lvl] = int(lines[lineIdx].transform(lvl))
        for r in range(image.rows):
            for c in range(image.cols):
                image.data[r][c] = lvls[image.data[r][c]]

        return image

    def addNoise(self):
        image = copy.deepcopy(self)
        for r in range(image.rows):
            for c in range(image.cols):
                randomInt = random.randint(0, 20)
                if randomInt == 0:
                    image.data[r][c] = 0
                elif randomInt == 20:
                    image.data[r][c] = 255
        return image

    def applyLinearFilter(self, filter: 'np.ndarray'):
        image = copy.deepcopy(self)
        data = np.array(image.data)
        n, m = filter.shape
        for r in range(image.rows):
            for c in range(image.cols):
                rstart = r - n // 2
                rend = r + n // 2
                cstart = c - m // 2
                cend = c + m // 2
                if rstart < 0 or rend >= image.rows or cstart < 0 or cend >= image.cols:
                    continue
                portion = data[rstart:rend + 1, cstart:cend + 1]
                pixel = int(np.sum(np.multiply(portion, filter)))
                image.data[r][c] = pixel

        return image

    def applyMedianFilter(self, n: 'int', m: 'int'):
        image = copy.deepcopy(self)
        data = np.array(image.data)
        for r in range(image.rows):
            for c in range(image.cols):
                rstart = r - n // 2
                rend = r + n // 2
                cstart = c - m // 2
                cend = c + m // 2
                if rstart < 0 or rend >= image.rows or cstart < 0 or cend >= image.cols:
                    continue
                block = data[rstart:rend + 1, cstart:cend + 1]
                pixel = int(np.median(block))
                image.data[r][c] = pixel

        return image
