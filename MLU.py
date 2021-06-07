# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:15:35 2021

@author: liush
"""

import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K
from keras.models import load_model
import random
import os
import matplotlib.pyplot as plt

img_rows, img_cols = 28, 28
batch_size = 48
num_classes = 10
epochs = 5
round_in_loop = 4 
rounds = 60 #communication round
freq = 1 #communication times in each loop

k, c = 20, 0.1 #total number of clients, fraction
m = k*c
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
x_train_all, y_train_all = [], []

path = "non_iid_data_mnist_range5_label/"
file_list = []
file_list = os.listdir(path)
for s in file_list:
    #print(s)
    a, b = s[7]+s[8], s[10]
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
for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    length[i] = len(x_train)
    length_all = length_all + length[i]
    weights.append(np.array(model.get_weights())) #initialize
set_es = [0]
global_model_test_loss = [] 
global_model_test_acc = [] 
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
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])
        #print('\n')
    if flag:
        weights_new = length[0]/length_all*weights[0]
        for i in range(1,k):
            weights_new = weights_new + length[i]/length_all*weights[i] # aggregate
        model.set_weights(weights_new)     # global model update
    else:
        weights_new = length[0]/length_all*weights[0]
        for i in range(1,k):
            weights_new = weights_new + length[i]/length_all*weights[i] # aggregate
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

plt.plot(global_model_test_acc, label='MLU')
plt.xlabel('round')
plt.ylabel('accuarcy')
plt.legend()
plt.show()