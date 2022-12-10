import numpy as np
import math
from operator import itemgetter
from linearTransformation import LinearTransformation
import random


class Image:
    def __init__(self, rows, cols, maxLevel) -> None:
        self.rows = rows
        self.cols = cols
        self.maxLevel = maxLevel

    def _histogram(self, mat):
        ret = np.zeros(self.maxLevel + 1)
        for r in range(self.rows):
            for c in range(self.cols):
                ret[mat[r][c]] += 1
        return ret
    
    def _probability(self, mat):
        # the probability is just the histogram divided by the number of pixels of an image
        # as the probability of each level is the number of occurrences of the level divided by the total number of pixels
        return self._histogram(mat) / (self.rows * self.cols)

    def _cumulatedHistogram(self, mat):
        hist = self._histogram(mat)
        return np.cumsum(hist)

    def _mean(self, mat):
        return np.mean(mat)

    def _variance(self, mat):
        mean = self._mean(mat)
        var = 0
        for r in range(self.rows):
            for c in range(self.cols):
                var += (mat[r][c] - mean) ** 2
        var /= self.rows * self.cols
        return var

    def _equalizeHist(self, mat):
        cumulatedHist = self._cumulatedHistogram(mat)
        lvls = [0] * (self.maxLevel + 1)
        n = self.rows * self.cols
        for i in range(self.maxLevel + 1):
            lvls[i] = math.floor(self.maxLevel / n * cumulatedHist[i])

        for r in range(self.rows):
            for c in range(self.cols):
                mat[r][c] = lvls[mat[r][c]]

    def _linearTransform(self, mat, points):
        points = sorted(points, key=itemgetter(0, 1))
        points.insert(0, (0, 0))
        points.append((self.maxLevel, self.maxLevel))
        lines = []
        for i in range(1, len(points)):
            if points[i - 1][0] == points[i][0]:
                raise Exception(
                    "two points cannot be on the same vertical line")
            lines.append(LinearTransformation(points[i - 1], points[i]))
        lvls = [0] * (self.maxLevel + 1)
        lineIdx = 0
        for lvl in range(self.maxLevel + 1):
            if lvl > points[lineIdx + 1][0]:
                lineIdx += 1
            lvls[lvl] = int(lines[lineIdx].transform(lvl))
        for r in range(self.rows):
            for c in range(self.cols):
                mat[r][c] = lvls[mat[r][c]]

    def _addNoise(self, mat):
        for r in range(self.rows):
            for c in range(self.cols):
                randomInt = random.randint(0, 20)
                if randomInt == 0:
                    mat[r][c] = 0
                elif randomInt == 20:
                    mat[r][c] = 255

    def _applyLinearFilter(self, mat, filter: 'np.ndarray'):
        data = np.array(mat)
        n, m = filter.shape
        for r in range(self.rows):
            for c in range(self.cols):
                if n % 2 == 0 or m % 2 == 0:
                    rstart = r
                    rend = r + n - 1
                    cstart = c
                    cend = c + m - 1
                else:
                    rstart = r - n // 2
                    rend = r + n // 2
                    cstart = c - m // 2
                    cend = c + m // 2
                if rstart < 0 or rend >= self.rows or cstart < 0 or cend >= self.cols:
                    continue
                portion = data[rstart:rend + 1, cstart:cend + 1]
                pixel = np.sum(np.multiply(portion, filter))
                pixel = int(np.clip(pixel, 0, self.maxLevel))
                mat[r][c] = pixel

    def _applyMedianFilter(self, mat, n: 'int', m: 'int'):
        data = np.array(mat)
        for r in range(self.rows):
            for c in range(self.cols):
                rstart = r - n // 2
                rend = r + n // 2
                cstart = c - m // 2
                cend = c + m // 2
                if rstart < 0 or rend >= self.rows or cstart < 0 or cend >= self.cols:
                    continue
                block = data[rstart:rend + 1, cstart:cend + 1]
                pixel = int(np.median(block))
                mat[r][c] = pixel

    def _otsu(self, mat):
        prob = self._probability(mat)
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
