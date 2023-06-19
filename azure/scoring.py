import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import os

def init():
    global sam, predictor
    checkpoint_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sam')
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    sam.to(device='cpu')

    predictor = SamPredictor(sam)

def run(image):
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    return image_embedding.tolist()
