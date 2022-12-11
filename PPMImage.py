import numpy as np
import pathlib
import cv2
from Image import Image
from PGMImage import PGMImage
from structuringElement import StructuringElement


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

    def __createImageFromMatrix(img: np.ndarray):
        rows, cols, _ = img.shape
        maxLevel = img.max()
        ppmImage = PPMImage(rows, cols, maxLevel)
        ppmImage.__r = img[:, :, 0]
        ppmImage.__g = img[:, :, 1]
        ppmImage.__b = img[:, :, 2]
        return ppmImage



    def convertImageToPPM(image: 'str | np.ndarray') -> 'PPMImage':
        """
        Create a PPMImage object from an image, the image argument can be a file path of the input image
        which can be a ppm, pgm, or other types of images, or a numpy array that represents the matrix of the image
        the matrix must hold the RGB values of the pixels of the image
        """
        if isinstance(image, str):
            extension = pathlib.Path(image).suffix
            if extension == ".ppm":
                return PPMImage.readFromFile(image)
            elif extension == ".pgm":
                pgmImage = PGMImage.readFromFile(image)
                data = pgmImage._PGMImage__data
                ppmImage = PPMImage(
                    pgmImage.rows, pgmImage.cols, pgmImage.maxLevel)
                ppmImage.__r = data
                ppmImage.__g = np.array(data)
                ppmImage.__b = np.array(data)
            else:
                img = cv2.imread(image)
                ppmImage = PPMImage.__createImageFromMatrix(img)
        else:
            # the image passed is a matrix here
            ppmImage = PPMImage.__createImageFromMatrix(image)

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

    def otsu(self):
        return self._otsu(self.__r), self._otsu(self.__g), self._otsu(self.__b)

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
    
    def mean(self):
        return self._mean(self.__r), self._mean(self.__g), self._mean(self.__b)

    def variance(self):
        return self._variance(self.__r), self._variance(self.__g), self._variance(self.__b)

    def histogram(self):
        return self._histogram(self.__r), self._histogram(self.__g), self._histogram(self.__b)

    def equalizeHistogram(self):
        self._equalizeHist(self.__r)
        self._equalizeHist(self.__g)
        self._equalizeHist(self.__b)
        return self
    
    def addNoise(self):
        self._addNoise(self.__r)
        self._addNoise(self.__g)
        self._addNoise(self.__b)
        return self

    def linearTransform(self, points):
        self._linearTransform(self.__r, points)
        self._linearTransform(self.__g, points)
        self._linearTransform(self.__b, points)
        return self
    
    def erode(self, se: StructuringElement):
        self._erode(self.__r, se)
        self._erode(self.__g, se)
        self._erode(self.__b, se)
        return self
    
    def dilate(self, se: StructuringElement):
        self._dilate(self.__r, se)
        self._dilate(self.__g, se)
        self._dilate(self.__b, se)
        return self

    def signalToNoiseRatio(original: 'PPMImage', treated: 'PPMImage'):
        return Image._signalToNoiseRatio(original.__r, treated.__r), Image._signalToNoiseRatio(original.__g, treated.__g), Image._signalToNoiseRatio(original.__b, treated.__b)
