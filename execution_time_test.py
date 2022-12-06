import cv2 as cv
from PPMImage import PPMImage
from time import time
import numpy as np

def display_exec_time():
    global t
    now = time()
    print(now - t)
    t = now

filepath = "./images/car.jpeg"
t = time()
image = PPMImage.convertImageToPPM(filepath)
display_exec_time()
kernel = [
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
]

image.applyLinearFilter(np.array(kernel))
display_exec_time()