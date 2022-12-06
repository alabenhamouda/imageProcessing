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
        occ = {}
        for r in range(self.rows):
            for c in range(self.cols):
                pixel = mat[r][c]
                if pixel not in occ:
                    occ[pixel] = 0
                occ[pixel] += 1
        return [occ.get(level, 0) for level in range(self.maxLevel + 1)]

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
