# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:21:20 2021

@author: liush
"""

import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.models import load_model
import random
import os
import math
import matplotlib.pyplot as plt

img_rows, img_cols = 28, 28
batch_size = 48
num_classes = 10
epochs = 5 #local model epoch
rounds = 60 #communication round
a = 1.36 # a = e/2, e = 2.72 
k, c = 20, 0.1 #total number of clients, fraction
m = k*c
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
x_train_all, y_train_all = [], []

path = "non_iid_data_mnist_range5_label/"
file_list = []
file_list = os.listdir(path)
for s in file_list:
    b = s[10]
    path_s = path + s
    if b == 'x':
        x_train_all.append(np.load(path_s))
    elif b=='y':
        y_train_all.append(np.load(path_s))
    
clients_index = []
for i in range(0,k):
    clients_index.append(i)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train =X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = load_model('w0.h5')#initialized model w0
weights = []#weights of each client
length_all = 0 #total size of data
length = np.zeros(shape = k) #size of each client
tw = np.zeros(shape = k) #time weighted value
global_model_weights = []
global_model_weights.append(model.get_weights())
for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    length[i] = len(x_train)
    length_all = length_all + length[i]
    weights.append(np.array(model.get_weights())) #initialize
length = length/length_all

global_model_test_loss = [] 
global_model_test_acc = []
t = np.zeros(shape = k) #choosen times of each client 
for r in range(0,rounds):
    s0 = random.sample(clients_index, int(m)) #clients of rounds k
    for i in range(0,int(m)):
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]
        t[s0[i]] = r + 1
        y_train = keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(global_model_weights[r]) #current local model
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(x_test, y_test))
 
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        score = model.evaluate(x_test, y_test, verbose=0)
    tw_all = 0 
    for i in range(0,k):
        if (r-t[i])>=2:
            tw[i] = math.pow(a, (t[i]-r+2))
        else:
            tw[i] = 1
        tw_all = tw_all + tw[i]
    tw = tw/tw_all
    ntw = length*tw
    ntw_all = 0
    for i in range(0,k):
        ntw_all = ntw_all + ntw[i]
    ntw = ntw/ntw_all
    #weights_new = ntw*weights
    weights_new = ntw[0]*weights[0]
    for i in range(1,k):
        weights_new = weights_new + ntw[i]*weights[i]# time weighted
    model.set_weights(weights_new)     # global model update
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    global_model_weights.append(model.get_weights())
    print ("round %d:"%(r+1),end = '\n')
    print('Global Model Test loss:', score[0])
    print('Global Model Test accuracy:', score[1])
    print('\n')
plt.plot(global_model_test_acc, label='TWF')
plt.xlabel('round')
plt.ylabel('accuarcy')
plt.legend()
plt.show()