# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:22:52 2021

@author: liush
"""

import numpy as np
import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import random
import os
import math
import matplotlib.pyplot as plt
from collections import Counter
import time
from function import plot_result 
import tensorflow
import pandas as pd
import time

start=time.time() 

img_rows, img_cols = 28, 28
batch_size = 48
num_classes = 10
epochs = 5
round_in_loop = 3 
a = 1.36 #a = e or e/2, e = 2.72 
rounds = 180 #communication round

k, c = 20, 0.4 #total number of clients, fraction
m = k*c
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

save_path = 'results_mnist_save/ALU+TWF+IWE-LN_1106_result1.xlsx'
model = load_model('w0_mnist.h5')#initialized model w0
#path = "non_iid_data_mnist_range5_label/"############################################
path = "data_mnist_noniid/"

###########################################################################consistency
from compute_RDM import compute_rdm
from compute_RDM import compute_representational_consistency
name_mnist_x_stimulus = 'data_mnist_stimulus/' + 'mnist' +'_x_stimulus.npy'
stimulus = np.load(name_mnist_x_stimulus)
#distance_measure = 'correlation', 'cosine', 'euclidean'
distance_measure = 'cosine'
global_rdm = compute_rdm(model, stimulus, distance_measure)#initialize global_rdm
flag_update_deeplayer = np.zeros(shape = (k,rounds))
representational_consistency_all = np.zeros(shape = (int(m), rounds))
save_path_flag = 'results_mnist_save/ALU+TWF+IWE-LN_thre_c04_1106_flag_1.xlsx'
#####################################################################


file_list = []
file_list = os.listdir(path)
for s in file_list:
    #print(s)
    b = s[10] #client_001_x_train.npy, client_001_x_test.npy, etc
    path_s = path + s
    if ((b == 'x') & (len(s)==21)):
        x_train_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==21)):
        y_train_all.append(np.load(path_s))
    elif ((b=='x') & (len(s)==20)):
        x_test_all.append(np.load(path_s))
    elif ((b=='y') & (len(s)==20)):
        y_test_all.append(np.load(path_s))  

start=time.time()   

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
    num_label[i] = len(Counter(y_train_all[i]))
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
    
    for i in range(0,int(m)):
        x_train = x_train_all[s0[i]]
        y_train = y_train_all[s0[i]]    
        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        t_g[s0[i]] = r + 1
        t_s[s0[i]] = r + 1
        
        model.set_weights(global_model_weights[r]) #current local model
        history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0)
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        #######################################################################################compute rdms
        local_rdm = compute_rdm(model, stimulus, distance_measure)  
        Pearson_coefficient_layer = compute_representational_consistency(local_rdm,global_rdm)
        representational_consistency_layer = Pearson_coefficient_layer[0]*Pearson_coefficient_layer[0]
        representational_consistency_all[i][r]= representational_consistency_layer
        #Pearson_coefficient_layer = compute_representational_consistency(local_rdm[j],global_rdm[j])
        #representational_consistency_layer = Pearson_coefficient_layer[0]*Pearson_coefficient_layer[0]
        #representational_consistency_all[i][j][r]= representational_consistency_layer
            #layer_weights_new.append(model.get_weights()[j])#layers of the trained model
            #layer_weights_old.append(global_model_weights[r][j])#layers of the current global model
        #layer_weights_new, layer_weights_old = np.array(layer_weights_new), np.array(layer_weights_old)
        #######################################################################################ALU
        #'''
        #'''
        t_p = r/rounds
        l_layer = 7
        if r==0:
            delta_acc = 0.1797
        elif r==1:
            delta_acc = global_model_test_acc[0]-0.1797
        else:
            delta_acc = global_model_test_acc[-1]-global_model_test_acc[-2]
        #threhold = 1/(1+np.exp(-(-0.00183699*t_p-0.260018*delta_acc+1.00128))) 
        threshold = 1/(1+np.exp(-(-0.626765*t_p-0.711312*delta_acc+2.52889))) 
        if(representational_consistency_all[i][r])>=threshold:
            flag_update_deeplayer[s0[i]][r] = 1 #update this layer
        else:
            flag_update_deeplayer[s0[i]][r] = random.choice((0,1))
            if flag_update_deeplayer[s0[i]][r] == 0: #not update this layer
                weights[s0[i]][4] = global_model_weights[-1][4]
        #'''
        ######################################################################################end

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
    for i in range(0,4):
        weights_new.append(weights_new_g[i])
    for i in range(4,5):
        weights_new.append(weights_new_s[i])
    for i in range(5,8):
        weights_new.append(weights_new_g[i])
    for j in range(0,4):
        for i in range(1,k):
            weights_new[j] = weights_new[j] + ntw_g[i]*weights[i][j]
    for j in range(4,5):    
        for i in range(1,k):
            weights_new[j] = weights_new[j] + ntw_s[i]*weights[i][j]
    for j in range(5,8):    
        for i in range(1,k):
            weights_new[j] = weights_new[j] + ntw_g[i]*weights[i][j]
    model.set_weights(weights_new)     # global model update
    
    global_model_weights.append(model.get_weights())
    print ("round %d:"%(r+1),end = '\n')
    score = model.evaluate(x_test, y_test, verbose=0)
    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    print('Global Model Test loss:', score[0])
    print('Global Model Test accuracy:', score[1])
    print('\n')
    
plot_result(global_model_test_acc,'ALU+TWF+IWE-LN','accuracy')
save_name = list(zip(global_model_test_acc, global_model_test_loss))
dataframe = pd.DataFrame(save_name, columns=['accuracy', 'loss'])
dataframe.to_excel(save_path, index=False)
##########################################################################save flag
save_name_flag = list(flag_update_deeplayer)
dataframe_flag = pd.DataFrame(save_name_flag)
dataframe_flag.to_excel(save_path_flag, index=False)
##############################################################################
end=time.time()
print('Running time: %d Seconds'%(end-start))