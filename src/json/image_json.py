import requests
import os
import numpy as np
from PIL import Image, ImageOps
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
IMAGE = []
DIR = os.path.dirname(__file__)+os.sep+"images"


def mutate_pixel_scale(image: np.ndarray):
    min = image.min()
    max = image.max()
    return 1 - (image - min)/(max-min)



for filename in os.listdir(DIR):
    image = Image.open(DIR+os.sep+filename).resize((28,28))
    image = ImageOps.grayscale(image)
    image = np.asarray(image)
    image = mutate_pixel_scale(image)
    plt.imshow(image, cmap='gray')
    plt.show()
    break

JSON = {
    'username': 'Hunter',
    'password': '1234',
    'image': IMAGE
}



