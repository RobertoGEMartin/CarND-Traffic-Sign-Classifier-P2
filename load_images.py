#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:26:41 2017

@author: Rober
"""

import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
import numpy as np
import scipy.misc as misc
from skimage import io
import os
import glob

path = "./web_images"

directory = "./web_images"
fs = glob.glob(directory + '/*.jpg')
X_custom = np.empty([0, 32, 32, 3], dtype = np.int32)

x_da = np.empty([0, 32, 32, 3], dtype = np.int32)
y_da = np.empty([0], dtype = np.int32)
if os.path.exists(directory):
    for i, file in enumerate(os.listdir(directory)):
        if not file.startswith('.'):
            label = os.path.splitext(file)[0]
            img = misc.imread(directory + '/' + file)
            x_da = np.append(x_da,  [img[:, :, :3]], axis = 0)
            y_da = np.append(y_da, label)

X_test_web = x_da

fig, axes = plt.subplots(figsize=[2.0, 2.0], ncols=5, nrows=1)
for i in range(5):
    image = X_test_web[i].squeeze()
    #Original image
    axes[0, i].imshow(image)


y_test_web = np.array(y_da)
print(y_test_web)

