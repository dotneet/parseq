import gradio as gr
from gradio import Brush
import os
import numpy as np
from PIL import ImageOps, Image

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

ckpt = os.environ.get("MODEL") or "./parseq.ckpt"
model = load_from_checkpoint(ckpt).eval().to("cpu")
img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

# 画像の二値化
def binarize(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    return image

def crop_to_bbox(image):
    padding = 20
    bbox = image.getbbox()
    if bbox:
        padded_bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
        width, height = image.size
        padded_bbox = (
            max(padded_bbox[0], 0),
            max(padded_bbox[1], 0),
            min(padded_bbox[2], width),
            min(padded_bbox[3], height)
        )
        image = image.crop(padded_bbox)
    return image


def predict(image_input):
    img = image_input['composite']
    img = crop_to_bbox(img)
    img = binarize(img)
    image = img.convert('RGB')
    image = ImageOps.invert(image)
    img_tensor = img_transform(image).unsqueeze(0)
    logits = model(img_tensor)
    pred = logits.softmax(-1)
    label, confidence = model.tokenizer.decode(pred)
    return label, image

with gr.Blocks() as demo:
    with gr.Column():
        im = gr.ImageEditor(
            type="pil",
            image_mode="RGBA",
            canvas_size=(1200, 480),
            scale=1,
            brush=Brush(default_color="red", default_size=3),
        )
        text = gr.Textbox()
        im_preview = gr.Image()
        im.change(predict, outputs=[
            text,
            im_preview,
            ],
            inputs=im,
            trigger_mode="always_last",
            show_progress="hidden")

def greet(name):
    return "Hello " + name + "!"

demo.launch() 