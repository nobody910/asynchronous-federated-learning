# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:37:48 2021

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
from collections import Counter
from math import log

img_rows, img_cols = 28, 28
batch_size = 48
num_classes = 10
epochs = 5
round_in_loop = 3 
a = 1.36 #a = e or e/2, e = 2.72 
rounds = 60 #communication round

k, c = 20, 0.1 #total number of clients, fraction
m = k*c
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
x_train_all, y_train_all = [], []

path = "non_iid_data_mnist_range5_label/"
file_list = []
file_list = os.listdir(path)
for s in file_list:
    #print(s)
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
global_model_weights = []
global_model_weights.append(model.get_weights())
weights = []#weights of each client
length_all = 0 #total size of data
length = np.zeros(shape = k) #size of each client
num_label_all = 0 #number of label of each client
num_label = np.zeros(shape = k)
for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    length[i] = len(x_train)
    length_all = length_all + length[i]
    element = dict(Counter(y_train_all[i]))
    infor_entropy = 0
    for label in element:
        ain = element[label]/length[i]
        infor_entropy = infor_entropy + ain*log(ain,2)*(-1)
    #num_label[i] = len(Counter(y_train_all[i]))
    num_label[i] = infor_entropy
    #num_label[i] = len(Counter(y_train_all[i]))
    num_label_all = num_label_all + num_label[i]
    weights.append(np.array(model.get_weights())) #initialize
length = length/length_all
num_label = num_label/num_label_all
set_es = [0]
#set_es = [0,11,12,13,14]#5/15
global_model_test_loss = [] 
global_model_test_acc = [] 
t_g = np.zeros(shape = k) #timestamp_general
t_s = np.zeros(shape = k) #timestamp_special
tw_g = np.zeros(shape = k) 
tw_s = np.zeros(shape = k)
for r in range(0,rounds):
    s0 = random.sample(clients_index, int(m)) #clients of rounds r
    
    if (r % round_in_loop) in set_es:
        flag = True
    else:
        flag = False
    for i in range(0,int(m)):
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = keras.utils.to_categorical(y_train, num_classes)
        if flag == True:
            t_g[s0[i]] = r + 1
            t_s[s0[i]] = r + 1
        else:
            t_g[s0[i]] = r + 1
        model.set_weights(global_model_weights[r]) #current local model
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(x_test, y_test))
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        score = model.evaluate(x_test, y_test, verbose=0)
    if flag:
        tw_g_all = 0
        tw_s_all = 0
        for i in range(0,k):
            tw_g[i] = math.pow(a,(t_g[i]-r-1))
            tw_s[i] = math.pow(a,(t_s[i]-r-1))
            tw_g_all = tw_g_all + tw_g[i]
            tw_s_all = tw_s_all + tw_s[i]
        tw_g = tw_g/tw_g_all
        tw_s = tw_s/tw_s_all
        ntw_g_all = 0
        ntw_s_all = 0
        ntw_g = length*tw_g*num_label
        ntw_s = length*tw_s*num_label
        for i in range(0,k):
            ntw_g_all = ntw_g_all + ntw_g[i]
            ntw_s_all = ntw_s_all + ntw_s[i]
        ntw_g = ntw_g/ntw_g_all
        ntw_s = ntw_s/ntw_s_all
        weights_new_g = ntw_g[0]*weights[0]
        weights_new_s = ntw_s[0]*weights[0]
        weights_new = []
        for i in range(0,5):
            weights_new.append(weights_new_g[i])
        for i in range(5,8):
            weights_new.append(weights_new_s[i])
        for j in range(0,5):
            for i in range(1,k):
                weights_new[j] = weights_new[j] + ntw_g[i]*weights[i][j]
        for j in range(5,8):    
            for i in range(1,k):
                weights_new[j] = weights_new[j] + ntw_s[i]*weights[i][j]
        model.set_weights(weights_new)     # global model update
    else:
        tw_g_all = 0
        for i in range(0,k):
            tw_g[i] = math.pow(a,(t_g[i]-r-1))
            tw_g_all = tw_g_all + tw_g[i]
        tw_g = tw_g/tw_g_all
        ntw_g_all = 0
        ntw_g = length*tw_g*num_label
        for i in range(0,k):
            ntw_g_all = ntw_g_all + ntw_g[i]
        ntw_g = ntw_g/ntw_g_all
        
        weights_new = ntw_g[0]*weights[0]
        for i in range(1,k):
            weights_new = weights_new + ntw_g[i]*weights[i]
        for i in range(5,8):
            weights_new[i] = global_model_weights[r][i]
        model.set_weights(weights_new)     # global model update
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_weights.append(model.get_weights())
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    print ("round %d:"%(r+1),end = '\n')
    print('Global Model Test loss:', score[0])
    print('Global Model Test accuracy:', score[1])
    print('\n')
plt.plot(global_model_test_acc, label='MLU+TWF+IWE-IE')
plt.xlabel('round')
plt.ylabel('accuarcy')
plt.legend()
plt.show()