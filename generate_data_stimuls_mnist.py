# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:47:23 2021

@author: liush
"""

#Generation of Local Data Sets.
#import keras
from keras.datasets import mnist
from keras import backend as K
import random
import numpy as np
import math

img_rows, img_cols = 28, 28# input image dimensions
num_classes = 10
num_per_class_stimulus = 5
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
idx_test_total = []
for i in range(0,len(y_test)):
    idx_test_total.append(i)
idx_test_local = []
idx_test = np.argsort(y_test) #sort,Index values from small to large
y_test_sorted = y_test[idx_test]
x_test_sorted = x_test[idx_test]
for i in range(num_classes):
    index_test_range =  np.argwhere(y_test_sorted == i)
    index_test_max = max(index_test_range)
    index_test_min = min(index_test_range)
    idx_test_local = idx_test_local+random.sample(
        idx_test_total[int(index_test_min):int(index_test_max)],num_per_class_stimulus )

x_test_local = x_test_sorted[idx_test_local] #local test data
#y_test_local = y_test_sorted[idx_test_local]

name_x_test = 'data_stimulus/' + 'mnist' +'_x_stimulus.npy'
    #name_y_test = 'data_stimulus/' + 'client_' + str(j) +'_y_test.npy'

#np.save(name_x_test, x_test_local)





    #np.save(name_y_test, y_test_local)

# client_01-50 :train:1000-2000; test:300-700
# client_51-60 :train:100-200; test:30-60