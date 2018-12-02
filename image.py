#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:36:52 2017

@author: Rober
"""

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from skimage import data, io, filters, color, util, exposure
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

from keras.preprocessing.image import ImageDataGenerator
import random

datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zca_whitening=True ,
            rescale=1,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False,
            fill_mode='nearest')

def genNewImages(x,label,count,folder):
    directory = './da/' + folder + '/' + str(label)
#    itemindex = files.selectimageswithlabel(y_train, label)
#    rand_index = random.choice(itemindex)
#    x = x_train[rand_index]
#    x = x.transpose(0,3, 1, 2)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=directory, save_prefix=str(label), 
                              save_format='jpeg'):
        i += 1
        if i > count:
            break  # otherwise the generator would loop indefinitely
    return



@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

#import cv2

#Normalize
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    return (x / 255.0).astype(np.float32)

#Using cv2
def cv2_rgb_to_hsv (images):
    hsv = []
    for img in images:
        hsv.append = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv

#Using tf
def tf_rgb_to_hsv(images):
    tensor_imgs = tf.image.rgb_to_hsv(images)
    x = tf.contrib.util.make_ndarray(tensor_imgs)
    return x

    
#Using %matplotlib
def rgb2hsv(images):
    '''
    matplotlib.colors.rgb_to_hsv(arr)
    convert float rgb values (in the range [0, 1]), in a numpy array to hsv values.
    Parameters: 
    arr : (..., 3) array-like All values must be in the range [0, 1]
    Returns: hsv : (..., 3) ndarray Colors converted to hsv values in range [0, 1]
    '''
    return rgb_to_hsv(images)

def preprocessing(imgs):
    imgs = color.rgb2grey(imgs)
    imgs = exposure.equalize_adapthist(imgs)
    return imgs

def showImagesInIndex(idx, x_train):
    col=4
    row=4
    fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
    for i in range(row):
        for j in range(row):
            rand_index = random.choice(idx)
            image = x_train[rand_index].squeeze()
            #Original image
            axes[i, j].imshow(image)
    return

def checkImages(x_train):
    col=4
    row=4
    fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
    for i in range(row):
        for j in range(row):
            rand_index = random.randint(0,len(x_train))
            image = x_train[rand_index].squeeze()
            #Original image
            axes[i, j].imshow(image)
    return
