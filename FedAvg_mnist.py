# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:04:38 2021

@author: liush
"""

import numpy as np
import tensorflow
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import random
import os
from function import plot_result 
import pandas as pd
import time

img_rows, img_cols = 28, 28
batch_size = 48
num_classes = 10#######################################################
epochs = 5
rounds = 180 #communication round

k, c = 20, 0.4 #total number of clients, fraction
m = k*c
(X_train, Y_train), (x_test, y_test) = mnist.load_data() #only need test set
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

save_path = 'results_mnist_save/FedAvg_c04_1103_result1.xlsx'
model = load_model('w0_mnist.h5')#initialized model w0
#path = "non_iid_data_mnist_range5_label/"############################################
path = "data_mnist_noniid/"

file_list = []
file_list = os.listdir(path)
for s in file_list:
    #print(s)
    b = s[10] #client_01_x_train.npy, client_01_x_test.npy, etc
    path_s = path + s
    if ((b == 'x') & (len(s)==21)):
        x_train_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==21)):
        y_train_all.append(np.load(path_s))
    elif ((b=='x') & (len(s)==20)):
        x_test_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==20)):
        y_test_all.append(np.load(path_s))  
    

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
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

start=time.time()

weights = []#weights of each client
length_all = 0 #total size of data
length = np.zeros(shape = k) #size of each client
for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    length[i] = len(x_train)
    length_all = length_all + length[i]
    weights.append(np.array(model.get_weights())) #initialize local model
    #weights[i] = weights[i].tolist()
    #weights[i] = np.array(weights[i])
global_model_weights = []
global_model_weights.append(model.get_weights())
'''
weights = [[None]*len(model.get_weights())]*k
for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    length[i] = len(x_train)
    length_all = length_all + length[i]
    weights_local = np.array(model.get_weights()) #initialize
    for j in range(len(weights_local)):
        weights[i][j] = weights_local[j]
    #weights[i] = np.array(weights[i])
#weights = np.array(weights)
'''
global_model_test_loss = [] 
global_model_test_acc = [] 
for r in range(0,rounds):
    s0 = random.sample(clients_index, int(m)) #clients of rounds r
    for i in range(0,int(m)):
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        model.set_weights(global_model_weights[r]) #current local model
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0
          )
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        #score = model.evaluate(x_test, y_test, verbose=0)
        #print('Local Model Test loss:', score[0])
        #print('Local Model Test accuracy:', score[1])
        #print('\n')
    weights_new = length[0]/length_all*weights[0]
    for i in range(1,k):
        weights_new = weights_new + length[i]/length_all*weights[i] # aggregate
    model.set_weights(weights_new)     # global model update
    global_model_weights.append(model.get_weights())
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    print ("round %d:"%(r+1),end = '\n')
    print('Global Model Test loss:', score[0])
    print('Global Model Test accuracy:', score[1])
    print('\n')

plot_result(global_model_test_acc,'FedAvg','accuracy')
save_name = list(zip(global_model_test_acc, global_model_test_loss))
dataframe = pd.DataFrame(save_name, columns=['accuracy', 'loss'])
dataframe.to_excel(save_path, index=False)
end=time.time()
print('Running time: %d Seconds'%(end-start))
#x_train = np.load('non-iid_data_minst/client_1_x_train.npy')
#y_train = np.load('non-iid_data_minst/client_1_y_train.npy')
# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)

'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.summary()#模型结构
weights = np.array(model.get_weights())#模型权重
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''