#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:45:32 2017

@author: Rober
"""

import files
import image
import random  
import numpy as np
#files.createFilesDA()

num_files = 2
num_labels = 4
y_final = np.empty((1,))
for label in range(num_labels):
   for i in range(num_files):  
      y_da = np.empty((num_files), dtype=np.uint8) 
      y_da[i] = label
      y_final = np.concatenate((y_final, y_da), axis=0)
y_final = y_final[1:]
             
                
#x_train, y_train, x_valid, y_valid, x_test, y_test = files.load_data()
#
#files.check_ds(x_train, y_train, x_valid, y_valid, x_test, y_test)
#
#print('-------------------\n')
#x_train_da, y_train_da, x_valid_da, y_valid_da, x_test_da, y_test_da = files.load_data_DA()
#files.check_ds(x_train_da, y_train_da, x_valid_da, y_valid_da, x_test_da, y_test_da)
#
#train_balance, _ = files.check_balance(y_train)
#valid_balance, _ = files.check_balance(y_valid)
#test_balance, _ = files.check_balance(y_test)

###
###
###labels = np.unique(y_train)
##
#files.removemagesinfolders(y_train)

def generator(x_train, y_train, train_balance, folder):
    files.createFolders(y_train, folder)
    ##TRAIN
    for i, new_samples in enumerate(train_balance):
        itemindex = files.selectimageswithlabel(y_train, i)
        rand_index = random.choice(itemindex)
        x = x_train[rand_index]
        x = x.transpose(0, 3, 1, 2)
        image.genNewImages(x,label=i,count=new_samples,folder=folder)
    return


#image.showImagesInIndex(itemindex, x_train)
#files.createFolders(y_train)

#labels = np.unique(y_train)
#for label in labels:
#    directory = './da/test/' + str(label)
#    fs = glob.glob(directory + '/*.jpeg')            
#    x_da = np.empty((len(fs),32,32,3))
#    y_da = np.empty((len(fs),1))
#    if os.path.exists(directory):
#        for i, f in enumerate(fs):
#            img = misc.imread(f)
#            x_da[i] = img
#            y_da[i] = label
# 
           

    
#imagelib.checkImages(stacked)
#
#itemindex = files.selectimageswithlabel(y_train, label)
#image.showImagesInIndex(itemindex, x_train)


