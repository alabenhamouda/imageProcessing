class PGMImage:

    def readFromFile(filepath):
        with open(filepath) as f:
            lines = f.readlines()
            image = PGMImage()
            image.type = lines[0].strip()
            i = 1
            if lines[i].startswith('#'):
                i += 1
            image.rows, image.cols = [int(x) for x in lines[i].split()]
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
            f.writelines([f'{self.rows} {self.cols}\n'])
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
