import numpy as np
import gradio as gr
from PPMImage import PPMImage
from PGMImage import PGMImage
import typing
import cv2 as cv


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

with gr.Blocks() as demo:
    with gr.Row():
        dropFile = gr.File()
        image = gr.Image()
        dropFile.change(fn=showImage, inputs=dropFile, outputs=image)
    with gr.Row():
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