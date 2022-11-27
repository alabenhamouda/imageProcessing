import numpy as np


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
