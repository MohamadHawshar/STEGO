# load all images in folder as rgb using PIL

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import color, morphology


# os list dir
train_path = "../Cleaned_aips/imgs/train/"
val_path = "../Cleaned_aips/imgs/val/"
train_imgs = []
val_imgs = []
train_destination_path = "./imgs/train/"
val_destination_path = "./imgs/val/"

for file in os.listdir(train_path):
    if file.endswith(".jpg"):
        train_imgs.append(file)

for file in os.listdir(val_path):
    if file.endswith(".jpg"):
        val_imgs.append(file)

for img in train_imgs:
    # load image
    image = Image.open(train_path + img)
    # convert to greyscale
    image = image.convert('L')
    # apply gaussian blur of alpha = 0.5
    image = gaussian_filter(image, sigma=0.5)
    # apply top hat filter radius = 30
    image = morphology.white_tophat(image,morphology.disk(30))
    # convert to rgb
    image = color.gray2rgb(image)
    # convert to Image
    image = Image.fromarray(image)
    # save image
    image.save(train_destination_path + img)

for img in val_imgs:
    # load image
    image = Image.open(val_path + img)
    # convert to greyscale
    image = image.convert('L')
    # apply gaussian blur of alpha = 0.5
    image = gaussian_filter(image, sigma=0.5)
    # apply top hat filter radius = 30
    image = morphology.white_tophat(image,morphology.disk(30))
    # convert to rgb
    image = color.gray2rgb(image)
    # convert to Image
    image = Image.fromarray(image)
    # save image
    image.save(val_destination_path + img)
