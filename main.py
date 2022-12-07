import numpy as np
import gradio as gr
from PPMImage import PPMImage
from PGMImage import PGMImage
import typing
import cv2 as cv
import constants
import helpers
import matplotlib

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

demo.launch()