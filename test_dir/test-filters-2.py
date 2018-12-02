#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:35:42 2017

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

#import image as imagelib
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage import data, io, filters, color, util, exposure
from skimage.filters import rank
from skimage.morphology import disk
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

def load_data():
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

    return X_train, y_train, X_valid, y_valid, X_test, y_test 

#Plot a matrix of images
col=5
row=5
fig, axes = plt.subplots(figsize=[7.0, 7.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image1 = exposure.rescale_intensity(image)
    axes[i, 1].imshow(image1, cmap="gray")
    #processed image
    image2 = exposure.equalize_hist(image)
    axes[i, 2].imshow(image2)
     #processed image
    image3 = color.rgb2hsv(image)
    image3 = exposure.equalize_hist(image3)
    image3 = color.hsv2rgb(image3)
    axes[i, 3].imshow(image3)
     #processed image
    image4 = exposure.equalize_adapthist(image,clip_limit=0.03)
    axes[i, 4].imshow(image4)

##TEST 2
col=5
row=5
fig, axes = plt.subplots(figsize=[7.0, 7.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    #processed image
    image1 = exposure.rescale_intensity(image)
    axes[i, 1].imshow(image1, )
    #processed image
    image2 = exposure.equalize_hist(image)
    axes[i, 2].imshow(image2)
     #processed image
    image3 = exposure.adjust_gamma(image)
    axes[i, 3].imshow(image3)
     #processed image
    selem = disk(30)
    image4 = rank.equalize(color.rgb2gray(image),selem=selem)
    axes[i, 4].imshow(image4,cmap="gray")

##TEST 3
col=5
row=5
fig, axes = plt.subplots(figsize=[7.0, 7.0], ncols=col, nrows=row)
fig.tight_layout()
list_labels = np.zeros((col,row))
for i in range(row):
    index = np.random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    #Original image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title("Original")
    #processed image
    #image1 = color.rgb2gray(image)
    image1 = sobel_each(image)
    image1 = exposure.rescale_intensity(1 - image1)
    axes[i, 1].imshow(image1, cmap="gray")
    axes[i, 1].set_title("Sobel in RGB")
    #processed image
    image2 = exposure.equalize_hist(image)
    axes[i, 2].imshow(image2)
    axes[i, 2].set_title("Hist Equa")
     #processed image
    image3 = exposure.adjust_gamma(image, 2)
    axes[i, 3].imshow(image3)
    axes[i, 3].set_title("Gamma adjust.")
     #processed image
    selem = disk(30)
    image4 = rank.equalize(color.rgb2gray(image),selem=selem)
    axes[i, 4].imshow(image4,cmap="gray")
    axes[i, 4].set_title("Local H. E.")

    
#Plot a matrix of images
col=8
row=8
fig, axes = plt.subplots(figsize=[12.0, 12.0], ncols=col, nrows=row)
list_labels = np.zeros((col,row))
for i in range(col):
    for j in range(row):
        index = np.random.randint(0, len(X_train))
        img = X_train[index].squeeze()
        img = imagelib.normalize(img)
        axes[i, j].imshow(img)
        list_labels[i,j] = y_train[index]

print("Labels: ")
print(list_labels)

####################
import random
def rgb2grey(x):
    return color.rgb2grey(x)
itemindex = np.argwhere(y_train==1)
X_label = X_train[itemindex,:,:,:]
#X_train_pp = exposure.rescale_intensity(X_train_pp, out_range=(0, 255))
X_train_pp = np.array(shape=(len(X_label),32,32))
for i in range(len(X_label)):
    img = X_label[i]
    img = color.rgb2grey(img)
    X_train_pp[i] = rank.equalize(img, selem=selem)
    
col=4
row=4
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
for i in range(row):
    for j in range(row):
        rand_index = random.choice(itemindex)
        image = X_train_pp[rand_index].squeeze()
        #Original image
        axes[i, j].imshow(image, cmap='grey')


#Convert to grayscale, e.g. single channel Y
itemindex = np.argwhere(y_train==1)
X_label = X_train[itemindex,:,:,:]
X = 0.299 * X_label[:, :, :, 0] + 0.587 * X_label[:, :, :, 1] + 0.114 * X_label[:, :, :, 2]
#Scale features to be in [0, 1]
X = (X / 255.0).astype(np.float32)

import warnings 

for i in range(X.shape[0]):
   with warnings.catch_warnings():
        warnings.simplefilter("ignore")a
        X[i] = exposure.equalize_adapthist(X[i])

col=4
row=4
fig, axes = plt.subplots(figsize=[8.0, 8.0], ncols=col, nrows=row)
for i in range(row):
    for j in range(row):
        rand_index = random.choice(itemindex)
        image = X[rand_index].squeeze()
        #Original image
        axes[i, j].imshow(image, cmap='grey')