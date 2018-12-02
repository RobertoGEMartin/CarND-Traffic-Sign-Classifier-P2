#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:46:15 2017

@author: Rober
"""
import image as imagelib
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

def load_data_DA():
# TODO: Fill this in based on where you saved the training and testing data

    training_file = './da/train-da.p'
    validation_file= './da/valid-da.p'
    testing_file = './da/test-da.p'
    
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
        
    X_train, y_train = train['x_da'], train['y_da']
    X_valid, y_valid = valid['x_da'], valid['y_da']
    X_test, y_test = test['x_da'], test['y_da']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def check_ds(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # TODO: Number of training examples
    n_train = len(X_train)
    
    # TODO: Number of testing examples.
    n_test = len(X_test)
    
    # TODO: What's the shape of an traffic sign image?
    image_shape = X_train[0].shape
    
    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))
    v_classes = np.unique(y_train)
    
    assert(len(X_train) == len(y_train))
    assert(len(X_valid) == len(y_valid))
    assert(len(X_test) == len(y_test))
    
    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    print("Values of classes =", v_classes)

    return

def check_balance(y_train):
    #plt.hist(y_train, bins=np.arange(y_train.min(), y_train.max()+1))
    count_bins = np.bincount(y_train) 
    print(count_bins)
    print('\nImbalance:')
    max_count_bin = count_bins.max()
    train_imbalance = []
    for i, c in enumerate(count_bins):
        diff = max_count_bin -c
        train_imbalance.append(diff)
        print('class: %2d >> Imb: %4d' % (i,diff))
    total_imbalance = sum(train_imbalance)
    print("----------------\nTotal Imbalance: %5d\n" % total_imbalance)
    return train_imbalance, total_imbalance

def selectimageswithlabel(y_train, label):
    itemindex = np.argwhere(y_train==label)
    #print(itemindex)
    return itemindex

import os
import glob
def createFolders(y_train, folder):
    folders = np.unique(y_train)
    for f in folders:
        directory = './da/' + folder + '/' + str(f)
        if not os.path.exists(directory):
            os.makedirs(directory)
            

def removemagesinfolders(y_train, folder):
    folders = np.unique(y_train)
    for f in folders:
        directory = './da/'  + folder + '/' + str(f)
        files = glob.glob(directory + '/*')
        if os.path.exists(directory):
            for f in files:
                os.remove(f)

from scipy import misc    
def readImagesInFolder(folder):
    #number of classes
    labels = 43
    for label in range(labels):
        directory = './da/' + folder + '/' + str(label)
        fs = glob.glob(directory + '/*.jpeg')
        x_da = np.empty((len(fs),32,32,3))
        y_da = np.empty((len(fs)))
        if os.path.exists(directory):
            for i, file in enumerate(fs):
                img = misc.imread(file)
                x_da[i] = img
                y_da[i] = label
        data = {'x_da':x_da, 'y_da':y_da}
        return data

def writetofile(data, file):
    pickle.dump( data, open( file, "wb" ) )


def createFilesDA():
    folders = ['train','valid','test']
    for folder in folders:
        data = readImagesInFolder(folder)
        filepath = './da/' + folder + '-da.p'
        writetofile(data,filepath)
    return

#################################
#######TEST
#################################
#directory = './da/' + str(1)
#files = glob.glob(directory + '/*.jpeg')            
#stacked = np.empty((len(files),32,32,3))
#if os.path.exists(directory):
#    for i, file in enumerate(files):
#        img = misc.imread(file)
#        stacked[i] = img
#
#imagelib.checkImages(stacked)

        
#################################
#data = {'x_da':[2,2,2], 'y_da':[1,2,3]}
#fileprueba = "./da/prueba.p"
#pickle.dump( data, open( fileprueba, "wb" ))
#
#with open(fileprueba, mode='rb') as f:
#    data2 = pickle.load(f)
#a = data2['x_da']
#b = data2['y_da']
    