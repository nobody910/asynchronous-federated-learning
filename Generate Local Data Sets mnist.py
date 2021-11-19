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

for j in range(10,61):###############################################################
    Nc = [2,3,4,5,6] # the number of classes in each local data set.
    Nc = Nc[random.randint(0,4)]
    labels = [0,1,2,3,4,5,6,7,8,9] #the names of classes
    classes = random.sample(labels, Nc) #name of the classes in this client 
    classes_weight = np.zeros(shape=10)
    sum_weight = 0
    
    for i in range(0,Nc):
        classes_weight[classes[i]] = random.random() #random of (0,1)
        sum_weight = sum_weight + classes_weight[classes[i]]
        
    Smin, Smax =100,2000 #the range of the local train data size
    num = random.randint(Smin,Smax) #the number of local train data size
    Pclasses = classes_weight/sum_weight*num #the train number of each classes
    
    Smin_test, Smax_test = 16, 330#the range of the local test data size
    num_test = random.randint(Smin_test,Smax_test) #the number of local test data size
    Pclasses_test = classes_weight/sum_weight*num_test #the test number of each classes
    
    for i in range(0,Nc):
        Pclasses[classes[i]] = math.floor(Pclasses[classes[i]]) #round down
        Pclasses_test[classes[i]] = math.floor(Pclasses_test[classes[i]]) #round down
        
    Pclasses = Pclasses.astype('int')
    Pclasses_test = Pclasses_test.astype('int')
    idx = np.argsort(y_train) #sort,Index values from small to large
    idx_test = np.argsort(y_test) #sort,Index values from small to large
    x_train_sorted = x_train[idx]
    y_train_sorted = y_train[idx]
    x_test_sorted = x_test[idx_test]
    y_test_sorted = y_test[idx_test]
    idx_total = []
    idx_test_total = []
    
    for i in range(0,60000):
        idx_total.append(i)
        
    for i in range(0,10000):
        idx_test_total.append(i)
        
    idx_local = []
    idx_test_local = []
    
    for i in range(0,Nc):
        index_range =  np.argwhere(y_train_sorted == classes[i])
        index_max = max(index_range)
        index_min = min(index_range)
        idx_local = idx_local+random.sample(
            idx_total[int(index_min):int(index_max)],Pclasses[classes[i]] )
        index_test_range =  np.argwhere(y_test_sorted == classes[i])
        index_test_max = max(index_test_range)
        index_test_min = min(index_test_range)
        idx_test_local = idx_test_local+random.sample(
            idx_test_total[int(index_test_min):int(index_test_max)],Pclasses_test[classes[i]] )

    x_train_local = x_train_sorted[idx_local] #local train data
    y_train_local = y_train_sorted[idx_local]
    x_test_local = x_test_sorted[idx_test_local] #local test data
    y_test_local = y_test_sorted[idx_test_local]

    name_x_train = 'data_mnist_noniid/' + 'client_' + str(j) +'_x_train.npy'
    name_y_train = 'data_mnist_noniid/' + 'client_' + str(j) +'_y_train.npy'
    name_x_test = 'data_mnist_noniid/' + 'client_' + str(j) +'_x_test.npy'
    name_y_test = 'data_mnist_noniid/' + 'client_' + str(j) +'_y_test.npy'

   
    np.save(name_x_train, x_train_local)################################################
    np.save(name_y_train, y_train_local)
    np.save(name_x_test, x_test_local)
    np.save(name_y_test, y_test_local)

# client_01-50 :train:1000-2000; test:300-700
# client_51-60 :train:100-200; test:30-60