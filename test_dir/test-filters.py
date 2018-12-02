#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:51:59 2017

@author: Rober
"""

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file= './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

#Plot a matrix of images
col=8
row=8
fig, axes = plt.subplots(figsize=[12.0, 12.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(col):
    for j in range(row):
        index = np.random.randint(0, len(X_train))
        image = X_train[index].squeeze()
        axes[i, j].imshow(image)
        list_labels[i,j] = y_train[index]

print("Labels: ")
print(list_labels)

plt.show()
#Plot histogram of labels
plt.hist(y_train, bins=np.arange(y_train.min(), y_train.max()+1))

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from matplotlib.colors import rgb_to_hsv
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
    imgs = normalize(imgs)
    imgs = rgb2hsv(imgs)
    return imgs

from skimage import data, io, filters

image = data.coins()
# ... or any other NumPy array!
edges = filters.sobel(image)
io.imshow(edges)
io.show()

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

from skimage import color, util, exposure

#Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image1 = color.rgb2grey(image)
    axes[i, 1].imshow(image1, cmap="gray")
    #processed image
    image2 = exposure.equalize_hist(image1)
    axes[i, 2].imshow(image2 , cmap="gray")
     #processed image
    image3 = exposure.adjust_gamma(image1)
    axes[i, 3].imshow(image3 , cmap="gray")
     #processed image
    image4 = exposure.equalize_adapthist(image1)
    axes[i, 4].imshow(image4 , cmap="gray")

from skimage import color, util, exposure

#Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image1 = color.rgb2lab(image)
    axes[i, 1].imshow(image1)
    #processed image
    image2 = exposure.equalize_hist(image1)
    axes[i, 2].imshow(image2)
     #processed image
    image3 = color.lab2rgb(image2)
    axes[i, 3].imshow(image3)
     #processed image
    image4 = color.rgb2grey(image3)
    axes[i, 4].imshow(image4, cmap="gray")

from skimage import color, util, exposure
from skimage import data
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)


#Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image1 = rescale_intensity(1 - sobel_each(image))
    axes[i, 1].imshow(image1)
    #processed image
    image2 = rescale_intensity(1 - sobel_hsv(image))
    axes[i, 2].imshow(image2 , cmap="gray")

#Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image1 = color.rgb2gray(image)
    axes[i, 1].imshow(image1, cmap="gray")
    #processed image
    image2 = exposure.equalize_hist(image1)
    axes[i, 2].imshow(image2 , cmap="gray")
    #processed image
    image3 = filters.sobel(image2)
    axes[i, 3].imshow(image3 , cmap="gray")
     #processed image
    image4 = rescale_intensity(1 - sobel_hsv(image2))
    axes[i, 4].imshow(image4 , cmap="gray")
    
    #Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image1 = color.rgb2hsv(image)
    axes[i, 1].imshow(image1)
    #processed image
    image2 = exposure.equalize_hist(image1)
    axes[i, 2].imshow(image2)
    #processed image
    image3 = sobel_each(image2)
    axes[i, 3].imshow(image3)
    #processed image
    image4 = color.hsv2rgb(image2)
    axes[i, 4].imshow(image4)
    
#Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image2 = exposure.equalize_hist(image)
    axes[i, 1].imshow(image2)
    #processed image
    image3 = exposure.equalize_adapthist(image)
    axes[i, 2].imshow(image3)

for i,img in enumerate(X_train[:10]):
   X_train[i] = exposure.equalize_adapthist(img)
   
#Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=32):
    col=1
    row=20
    fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
    list_labels = np.zeros((col,row))
    for i in range(row):
        image = X_batch[i].squeeze()
        #Original image
        axes[i, 0].imshow(image)
    break


####
from skimage import color, util, exposure
import matplotlib.pyplot as plt


col=5
row=5
fig, axes = plt.subplots(col, row,figsize=[8.0, 8.0])
list_labels = np.zeros((col,row))
for i in range(row):
    for j in range(col):
        index = np.random.randint(0, len(X_train))
        image = X_train[index].squeeze()
        #Original image
        axes[i, j].imshow(image)
        #processed image
        image1 = color.rgb2gray(image)
        axes[i, j].imshow(image1, cmap="gray")
        #processed image
        image2 = exposure.equalize_hist(image1)
        axes[i, j].imshow(image2, cmap="gray")
        #processed image
        image3 = exposure.equalize_hist(image1)
        axes[i, j].imshow(image3, cmap="gray")
        #processed image, 
        image4 = exposure.equalize_hist(image1)
        axes[i, j].imshow(image4, cmap="gray")

col=5
row=5
index = np.zeros(shape=col)
fig, axes = plt.subplots(col, row, figsize=[8.0, 8.0])
list_labels = np.zeros((col,row))
images = []
for i in range(row):
    index = np.random.randint(0, len(X_train))
    for j in range(col):
        image = X_train[index].squeeze()
        #Original image
        axes[i, j].imshow(image)
        #processed image
        image1 = color.rgb2gray(image)
        axes[i, j].imshow(image1)

import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
import numpy as np
import scipy.misc

imgs, img_rs,  = [], []
y_test_web = []
i_width, i_height = 32, 32
path = "./web_images"
for f in os.listdir(path):
    if not f.startswith('.'):
        ext = os.path.splitext(f)[1]
        label = os.path.splitext(f)[0]
        y_test_web.append(int(label))
        pix = np.array(Image.open(os.path.join(path,f)))
        imgs.append(pix)
        img_rs.append(scipy.misc.imresize(pix, (i_height, i_width)))

print(y_test_web)
y_test_web = np.array(y_test_web)
print(y_test_web)

for i in range(5):
    plt.figure(figsize=[2.0, 2.0])
    plt.imshow(imgs[i].squeeze())
    plt.figure(figsize=[2.0, 2.0])
    plt.imshow(img_rs[i].squeeze())
    
#One-hot encode
from sklearn.preprocessing import LabelBinarizer
n_classes = len(np.unique(y_train))
encoder = LabelBinarizer()
labels = np.arange(n_classes)
encoder.fit(labels)

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    out = encoder.transform(x).astype(np.float32)
    return out

print(y_train[0])
y_train = one_hot_encode(y_train)
y_valid = one_hot_encode(y_valid)
y_test = one_hot_encode(y_test)
print(y_train[0])
    
y_test_web = one_hot_encode(y_test_web)
print(y_test_web[0])
