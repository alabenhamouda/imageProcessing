import numpy as np
import gradio as gr
from PPMImage import PPMImage
from PGMImage import PGMImage
import typing


def showImage(file: 'typing.TextIO'):
    if file is not None:
        image = PPMImage.convertImageToPPM(file.name)

        return image[:, :]


def applyFilter(file: 'typing.TextIO', kernel: 'np.ndarray'):
    if file is None:
        return
    kernel = kernel.astype(dtype=float)
    image = PPMImage.convertImageToPPM(file.name)
    image.applyLinearFilter(kernel)
    return image[:, :]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            dropFile = gr.File()
            image = gr.Image()
            dropFile.change(fn=showImage, inputs=dropFile, outputs=image)
            kernel = gr.DataFrame(type="numpy", headers=None)
            applyButton = gr.Button()
        with gr.Column():
            outputImage = gr.Image()
            applyButton.click(fn=applyFilter, inputs=[
                              dropFile, kernel], outputs=outputImage)

demo.launch()
