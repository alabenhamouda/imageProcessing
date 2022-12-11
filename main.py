import numpy as np
import gradio as gr
from PPMImage import PPMImage
from PGMImage import PGMImage
import typing
import cv2 as cv
import constants
import helpers
import matplotlib
import matplotlib.pyplot as plt
from operator import itemgetter
from structuringElement import StructuringElement

matplotlib.use("Agg")


def showImage(file: 'typing.TextIO'):
    if file is not None:
        image = PPMImage.convertImageToPPM(file.name)

        return image[:, :]


def applyFilter(file: 'typing.TextIO', kernel: 'np.ndarray'):
    if file is None:
        return
    kernel = kernel.astype(dtype=float)
    # image = PPMImage.convertImageToPPM(file.name)
    # image.applyLinearFilter(kernel)
    image = cv.imread(file.name)
    filtered = cv.filter2D(image, -1, kernel)
    return filtered

def applyMedianFilter(file: 'typing.TextIO'):
    if file is None:
        return
    # image = PPMImage.convertImageToPPM(file.name)
    # image.applyMedianFilter(5, 5)
    image = cv.imread(file.name)
    filtered = cv.medianBlur(image, 5)
    return filtered
    
def equalizeHist(file: typing.TextIO):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    image.equalizeHistogram()
    return image[:,:]

def onDropdownChange(file: typing.TextIO, dropdown_item):
    if file is None:
        return gr.Dataframe.update(visible=False), gr.Plot.update(visible=False), gr.Plot.update(visible=False), gr.Plot.update(visible=False)
    image = PPMImage.convertImageToPPM(file.name)
    if dropdown_item == constants.MEAN:
        mean_r, mean_g, mean_b = image.mean()
        values = [
            ["red", mean_r],
            ["green", mean_g],
            ["blue", mean_b]
        ]
        return gr.Dataframe.update(visible=True, value=values), gr.Plot.update(visible=False), gr.Plot.update(visible=False), gr.Plot.update(visible=False)
    elif dropdown_item == constants.VARIANCE:
        variance_r, variance_g, variance_b = image.variance()
        values = [
            ["red", variance_r],
            ["green", variance_g],
            ["blue", variance_b]
        ]
        return gr.Dataframe.update(visible=True, value=values), gr.Plot.update(visible=False), gr.Plot.update(visible=False), gr.Plot.update(visible=False)
    elif dropdown_item == constants.HISTOGRAM:
        hist_r, hist_g, hist_b = image.histogram()
        plot_r, plot_g, plot_b = [plot[0] for plot in [helpers.plot_histogram(hist) for hist in [hist_r, hist_g, hist_b]]]
        plot_r.suptitle("Histogram of the red intensity level")
        plot_g.suptitle("Histogram of the green intensity level")
        plot_b.suptitle("Histogram of the blue intensity level")
        return gr.Dataframe.update(visible=False), gr.Plot.update(visible=True, value=plot_r), gr.Plot.update(visible=True, value=plot_g), gr.Plot.update(visible=True, value=plot_b)

def addNoise(file: typing.TextIO):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    image.addNoise()
    return image[:,:]

def linearTransformation(file: typing.TextIO, points):
    if file is None:
        return None
    try:
        points = points.astype(float)
    except:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    image.linearTransform(points)
    return image[:,:]

linear_transform_fig, linear_transform_ax = plt.subplots()

def plotTransformationLines(points: np.ndarray):
    try:
        points = points.astype(int)
    except:
        return None
    points = np.insert(points, 0, [[0, 0]], axis=0)
    points = np.append(points, [[255, 255]], axis=0)
    points = np.array(sorted(points, key=itemgetter(0, 1)))
    helpers.plot_points(linear_transform_ax, points)
    return linear_transform_fig

def onThresholdChange(type):
    if type == constants.NORMALTHRESHOLDING:
        return gr.DataFrame.update(visible=True), gr.Number.update(visible=False)
    else:
        return gr.DataFrame.update(visible=False), gr.Number.update(visible=True)


def thresholdImage(file, thresholds, threshold, type):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    if type == constants.NORMALTHRESHOLDING:
        try:
            thresholds = thresholds.astype(int)
        except:
            return None
        image.rgbThreshold(*thresholds[0])
    elif type == constants.ANDTHRESHOLDING:
        image.andThreshold(threshold)
    elif type == constants.ORTHRESHOLDING:
        image.orThreshold(threshold)
    return image[:,:]

def calculateOtsuThresholds(file: typing.TextIO):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    tr, tg, tb = image.otsu()
    # update the threshold type to be normal thresholding and set the thresholds value on the table
    return gr.DataFrame.update(value=[[tr, tg, tb]]), gr.Dropdown.update(value=constants.NORMALTHRESHOLDING)

def erodeImage(file: typing.TextIO, kernel: np.ndarray, seed: np.ndarray):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    # image = cv.imread(file.name)
    try:
        kernel = kernel.astype(int)
        seed = seed.astype(int)
        seed = tuple(seed[0])
    except:
        return None
    image.erode(StructuringElement(kernel, seed))
    # image = cv.erode(image, kernel, iterations=1)
    return image[:,:]
    # return image

def dilateImage(file: typing.TextIO, kernel: np.ndarray, seed: np.ndarray):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    # image = cv.imread(file.name)
    try:
        kernel = kernel.astype(int)
        seed = seed.astype(int)
        seed = tuple(seed[0])
    except:
        return None
    image.dilate(StructuringElement(kernel, seed))
    # image = cv.dilate(image, kernel, iterations=1)
    return image[:,:]
    # return image

def openImage(file: typing.TextIO, kernel: np.ndarray, seed: np.ndarray):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    try:
        kernel = kernel.astype(int)
        seed = seed.astype(int)
        seed = tuple(seed[0])
    except:
        return None
    se = StructuringElement(kernel, seed)
    image.erode(se)
    image.dilate(se)
    return image[:,:]

def closeImage(file: typing.TextIO, kernel: np.ndarray, seed: np.ndarray):
    if file is None:
        return None
    image = PPMImage.convertImageToPPM(file.name)
    try:
        kernel = kernel.astype(int)
        seed = seed.astype(int)
        seed = tuple(seed[0])
    except:
        return None
    se = StructuringElement(kernel, seed)
    image.dilate(se)
    image.erode(se)
    return image[:,:]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            dropFile = gr.File()
        with gr.Column():
            image = gr.Image()
        dropFile.change(fn=showImage, inputs=dropFile, outputs=image)
    with gr.Row():
        with gr.Tab("general informations"):
            with gr.Row():
                with gr.Column():
                    variable_to_compute = gr.Radio(choices=[constants.MEAN, constants.VARIANCE, constants.HISTOGRAM], label="variable to compute")
                    recompute = gr.Button("Recompute")
                with gr.Column():
                    values = gr.Dataframe(visible=False, headers=["intensity level", "value"], interactive=False)
                    plot_r = gr.Plot(visible=False)
                    plot_g = gr.Plot(visible=False)
                    plot_b = gr.Plot(visible=False)
                    variable_to_compute.change(fn=onDropdownChange, inputs=[dropFile, variable_to_compute], outputs=[values, plot_r, plot_g, plot_b])
                    recompute.click(fn=onDropdownChange, inputs=[dropFile, variable_to_compute], outputs=[values, plot_r, plot_g, plot_b])
        with gr.Tab("add noise"):
            with gr.Row():
                with gr.Column():
                    applyButton = gr.Button("Add noise")
                with gr.Column():
                    outputImage = gr.Image()
                    applyButton.click(fn=addNoise, inputs=dropFile, outputs=outputImage)
        with gr.Tab("linear filter"):
            with gr.Row():
                with gr.Column():
                    exampleKernel = [
                        [-1, -1, -1],
                        [-1, 9, -1],
                        [-1, -1, -1]
                    ]
                    kernel = gr.DataFrame(type="numpy", headers=None, value=exampleKernel)
                    applyButton = gr.Button()
                with gr.Column():
                    outputImage = gr.Image()
                    applyButton.click(fn=applyFilter, inputs=[
                                    dropFile, kernel], outputs=outputImage)
        with gr.Tab("median filter"):
            with gr.Row():
                with gr.Column():
                    applyButton = gr.Button()
                with gr.Column():
                    outputImage = gr.Image()
                    applyButton.click(fn=applyMedianFilter, inputs=dropFile, outputs=outputImage)
        with gr.Tab("Equalize histogram"):
            with gr.Row():
                with gr.Column():
                    applyButton = gr.Button("Equalize histogram")
                with gr.Column():
                    outputImage = gr.Image()
                    applyButton.click(fn=equalizeHist, inputs=dropFile, outputs=outputImage)
        with gr.Tab("linear transformation"):
            with gr.Row():
                with gr.Column():
                    points = gr.DataFrame(headers=["X", "Y"], col_count=(2, 'fixed'), type="numpy", datatype="number")
                    plot = gr.Plot()
                    points.change(fn=plotTransformationLines, inputs=points, outputs=plot)
                    applyButton = gr.Button("Apply linear transformation")
                with gr.Column():
                    outputImage = gr.Image()
                    applyButton.click(fn=linearTransformation, inputs=[dropFile, points], outputs=outputImage)
        with gr.Tab("Threshold image"):
            with gr.Row():
                with gr.Column():
                    thresholding_types = [constants.NORMALTHRESHOLDING, constants.ANDTHRESHOLDING, constants.ORTHRESHOLDING]
                    thresholding_type = gr.Dropdown(choices=thresholding_types, value=constants.NORMALTHRESHOLDING)
                    thresholds = gr.DataFrame(headers=["Red", "Green", "Blue"], col_count=(3, 'fixed'), row_count=(1, 'fixed'), type="numpy", datatype="number")
                    threshold = gr.Number(label="threshold", visible=False)
                    thresholding_type.change(fn=onThresholdChange, inputs=thresholding_type, outputs=[thresholds, threshold])
                    calculate_otsu = gr.Button("Calculate otsu thresholds")
                    calculate_otsu.click(fn=calculateOtsuThresholds, inputs=dropFile, outputs=[thresholds, thresholding_type])
                    applyButton = gr.Button("Apply thresholding on image")
                with gr.Column():
                    outputImage = gr.Image()
                    applyButton.click(fn=thresholdImage, inputs=[dropFile, thresholds, threshold, thresholding_type], outputs=outputImage)
        with gr.Tab("morphological operations"):
            with gr.Row():
                with gr.Column():
                    exampleKernel = [
                        [1, 1],
                        [1, 1]
                    ]
                    kernel = gr.DataFrame(type="numpy", headers=None, value=exampleKernel)
                    seed = gr.DataFrame(headers=["X", "Y"], col_count=(2, 'fixed'), row_count=(1, 'fixed'), type="numpy", datatype="number")
                    erodeButton = gr.Button("Erode image")
                    dilateButton = gr.Button("Dilate image")
                    openButton = gr.Button("Open image")
                    closeButton = gr.Button("closeImage")
                with gr.Column():
                    outputImage = gr.Image()
                    erodeButton.click(fn=erodeImage, inputs=[dropFile, kernel, seed], outputs=outputImage)
                    dilateButton.click(fn=dilateImage, inputs=[dropFile, kernel, seed], outputs=outputImage)
                    openButton.click(fn=openImage, inputs=[dropFile, kernel, seed], outputs=outputImage)
                    closeButton.click(fn=closeImage, inputs=[dropFile, kernel, seed], outputs=outputImage)

demo.launch()