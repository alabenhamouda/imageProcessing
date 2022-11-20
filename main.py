from Image import PGMImage
filepath = './images/balloons.ascii.pgm'
balloon = PGMImage.readFromFile(filepath=filepath)
balloon.writeToFile("balloon.pgm")
