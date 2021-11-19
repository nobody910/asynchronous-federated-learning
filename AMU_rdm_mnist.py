# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:15:35 2021

@author: liush
"""

import numpy as np
import tensorflow.compat.v1.keras
from tensorflow.compat.v1.keras.datasets import mnist
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import load_model
import random
import os
import matplotlib.pyplot as plt
from function import plot_result 
import tensorflow
import pandas as pd
import time
from scipy.spatial.distance import pdist

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # 使用第一, 三块GPU
#tensorflow.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.99)
#gpu_options = tensorflow.compat.v1.GPUOptions(allow_growth=True)
#sess = tensorflow.compat.v1.Session(config=tensorflow.compat.v1.ConfigProto(gpu_options=gpu_options))  
tensorflow.compat.v1.disable_eager_execution()
img_rows, img_cols = 28, 28
batch_size = 48
num_classes = 10
epochs = 5
round_in_loop = 3 
rounds = 180 #communication round
freq = 1 #communication times in each loop

k, c = 20, 0.4 #total number of clients, fraction
m = k*c
(X_train, Y_train), (x_test, y_test) = mnist.load_data()
x_train_all, y_train_all, x_test_all, y_test_all = [], [], [], []

save_path = 'results_mnist_save/ALU_c04_1106_thre_result1.xlsx'
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
save_path_flag = 'results_mnist_save/ALU_c04_thre_1106_flag_1.xlsx'
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
for i in range(0,k):
    x_train = x_train_all[i]
    y_train = y_train_all[i]
    length[i] = len(x_train)
    length_all = length_all + length[i]
    weights.append(np.array(model.get_weights())) #initialize
set_es = [0]
global_model_test_loss = [] 
global_model_test_acc = [] 
#layer_dist_euclidean_all = np.zeros(shape = (8,rounds))
#layer_dist_cosine_all = np.zeros(shape = (8,rounds))

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
          verbose=0)
    #model.summary() #model structure
    #weights = np.array(model.get_weights())
        weights[s0[i]] = np.array(model.get_weights()) #local model weights update
        #layer_weights_new, layer_weights_old = [], []
        #layer_dist_euclidean = np.zeros(shape = 8)
        #layer_dist_cosine = np.zeros(shape = 8)
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
        t_p = r/rounds
        l_layer = 7
        if r==0:
            delta_acc = 0.1797
        elif r==1:
            delta_acc = global_model_test_acc[0]-0.1797
        else:
            delta_acc = global_model_test_acc[-1]-global_model_test_acc[-2]
        #threhold = 1/(1+np.exp(-(-0.00183699*t_p-0.260018*delta_acc+1.00128))) 
        threshold = 1.00173/(1+np.exp(-(-0.30594*t_p-30.0952*delta_acc+6.52589))) 
        if(representational_consistency_all[i][r])>=threshold:
            flag_update_deeplayer[s0[i]][r] = 1 #update this layer
        else:
            flag_update_deeplayer[s0[i]][r] = random.choice((0,1))
            if flag_update_deeplayer[s0[i]][r] == 0:#not update this layer
                weights[s0[i]][4] = global_model_weights[-1][4]
        #'''
        ######################################################################################end
         #for j in range(8):
            #layer_weights_new[j] = layer_weights_new[j].reshape(-1,layer_weights_new[j].size)
            #layer_weights_old[j] = layer_weights_old[j].reshape(-1,layer_weights_old[j].size)
            #layer_dist_euclidean = np.linalg.norm(layer_weights_new[j] - layer_weights_old[j])
            #layer_dist_cosine = pdist(np.vstack([layer_weights_new[j],layer_weights_old[j]]),'cosine')
            #layer_dist_euclidean_all[j][r] = layer_dist_euclidean
            #layer_dist_cosine_all[j][r] = layer_dist_cosine
        
        #score = model.evaluate(x_test, y_test, verbose=0)
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])
        #print('\n')
    weights_new = length[0]/length_all*weights[0]
    for i in range(1,k):
        weights_new = weights_new + length[i]/length_all*weights[i] # aggregate
    model.set_weights(weights_new)     # global model update
    global_model = model
    global_rdm = compute_rdm(global_model, stimulus, distance_measure)
        
    global_model_weights.append(model.get_weights())    
    print ("round %d:"%(r+1),end = '\n')
    score = model.evaluate(x_test, y_test, verbose=0)

    global_model_test_loss.append(score[0])
    global_model_test_acc.append(score[1])
    print('Global Model Test loss:', score[0])
    print('Global Model Test accuracy:', score[1])
    print('\n')

plot_result(global_model_test_acc,'ALU','accuracy')
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
###################################################################curve fit
'''
re_dense = representational_consistency_all
re_dense_median = np.median(re_dense, axis=0)
plt.plot(re_dense_median)

from scipy.optimize import curve_fit
t_p = np.linspace(1,rounds,rounds)/rounds
l_layer =3.0*np.ones(shape =rounds)
acc_2 = global_model_test_acc
acc_1 = np.zeros(shape = rounds)
acc_1[0] = 0.1797
for i in range(1,rounds):
    acc_1[i] = global_model_test_acc[i-1]
delta_acc = acc_2 - acc_1
y = re_dense_median
X = (t_p,delta_acc)
#a1,a2,a3,a4=1,1,1,1
#p0=1,1,1,1
def func1(X,a1,a3,a4):
    t_p,ldelta_acc = X
    #return a1*t_p+a3*delta_acc+a4#线性拟合
    return 1/(1+np.exp(-(a1*t_p+a3*delta_acc+a4))) #logit拟合
    ##return 1/(1+np.exp(-(a1*X[0]+a2*X[1]+a3*X[2]+a4))) 
# popt返回的是给定模型的最优参数。我们可以使用pcov的值检测拟合的质量，其对角线元素值代表着每个参数的方差。
popt, pcov = curve_fit(func1, (t_p, delta_acc),y)
a,b,c = popt[0],popt[1],popt[2]
yvals = func1(X,a,b,c) #拟合y值
error = yvals-y
plt.plot(y)
plt.plot(yvals)
#logit:threshold = 1.00173/(1+np.exp(-(-0.30594*t_p-30.0952*delta_acc+6.52589))) 
#线性：threshold = 1/(1+np.exp(-(-0.00183699*t_p-0.260018*delta_acc+1.00128))) 

#re_6, re_7 = np.zeros(shape=(int(m),rounds)), np.zeros(shape=(int(m),rounds))
#for i in range(int(m)):
#    re_6[i] = representational_consistency_all[i][0]
#    re_7[i] = representational_consistency_all[i][1]
re_6_median = np.median(re_6, axis=0)
re_7_median = np.median(re_7, axis=0)
plt.plot(re_6_median)
plt.plot(re_7_median)
name_layer6_median = 'representational_consistency/' + 'mnist' +'_layer6_median-1.npy'
np.save(name_layer6_median, re_6_median)
name_layer7_median = 'representational_consistency/' + 'mnist' +'_layer7_median-1.npy'
np.save(name_layer7_median, re_7_median)

from scipy.optimize import curve_fit
t_p = np.linspace(1,100,100)/100
l_layer =3.0*np.ones(shape =100)
acc_2 = global_model_test_acc
acc_1 = np.zeros(shape = 100)
acc_1[0] = 0.1797
for i in range(1,100):
    acc_1[i] = global_model_test_acc[i-1]
delta_acc = acc_2 - acc_1
y = re_6_median
X = (t_p,delta_acc)
#a1,a2,a3,a4=1,1,1,1
#p0=1,1,1,1
def func1(X,a1,a3,a4):
    t_p,ldelta_acc = X
    return a1*t_p+a3*delta_acc+a4
    #return 1/(1+np.exp(-(a1*t_p+a3*delta_acc+a4))) 
    ##return 1/(1+np.exp(-(a1*X[0]+a2*X[1]+a3*X[2]+a4))) 
# popt返回的是给定模型的最优参数。我们可以使用pcov的值检测拟合的质量，其对角线元素值代表着每个参数的方差。
popt, pcov = curve_fit(func1, (t_p, delta_acc),y)
a,b,c= popt[0],popt[1],popt[2]
yvals = func1(X,a,b,c) #拟合y值

error = yvals-y
#threhold = 1/(1+np.exp(-(0.235096*t_p+2.39499*l_layer+-7.52575*delta_acc+-14.7663))) 

l_layer_2 =8.0*np.ones(shape =100)
y2 = re_7_median
def func2(X,a1,a3,a4):
    t_p,delta_acc = X
    return a1*t_p+a3*delta_acc+a4 
X = (t_p,delta_acc)
popt2, pcov2 = curve_fit(func2, (t_p, delta_acc),y2)
a2,b2,c2= popt2[0],popt2[1],popt2[2]
yvals2 = func2(X,a2,b2,c2) #拟合y值
error2 = yvals2-y2
#threhold = 0.345955*t_p+-0.320097*delta_acc+0.212467
'''

