# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 12:08:45 2021

@author: ls
"""
import numpy as np
from tensorflow.keras.models import Model
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

#name_mnist_x_stimulus = 'data_stimulus/' + 'mnist' +'_x_stimulus.npy'
#stimulus = np.load(name_mnist_x_stimulus)
#distance_measure = 'correlation', 'cosine', 'euclidean'
#distance_measure = 'euclidean'

def compute_rdm(model, stimulus, distance_measure):
    #mnist_stimulus = np.load(name_mnist_x_stimulus)
    #rdms_all_layers = np.empty((2, len(stimulus), len(stimulus)))
    #rdms = np.empty((len(model.layers),  len(stimulus), len(stimulus)))
    #rdms_all_layers = np.empty(( len(stimulus), len(stimulus)))
    #for i in range(len(model.layers)-2, len(model.layers)):
    layer_model = Model(inputs=model.input, outputs=model.layers[4].output)
    feature_all = []
    for j in range(len(stimulus)):
        x_stimulu= np.expand_dims(stimulus[j],axis=0)
        feature=layer_model.predict(x_stimulu)
        feature = np.array(feature).reshape(-1,feature.size)
        feature_all.append(feature)
    feature_all = np.array(feature_all)
    rdms_one_layer = squareform(pdist(feature_all.reshape(len(stimulus), -1), distance_measure))
    #rdms_all_layers[i-6] = rdms_one_layer
    return rdms_one_layer   
    #return rdms_all_layers

def compute_representational_consistency(rdm1,rdm2):
    rdm_tri_vercor_1, rdm_tri_vercor_2= [], []
    for i in range(len(rdm1)):
        for j in range(len(rdm1)):
            if j>i:
                rdm_tri_vercor_1.append(rdm1[i][j])
                rdm_tri_vercor_2.append(rdm2[i][j])
    pccs = pearsonr(rdm_tri_vercor_1, rdm_tri_vercor_2)
    return pccs

'''
#from scipy.stats import pearsonr
#tri = np.triu(rdm[0],0)
rdm_tri_vercor_01 = []
for i in range(len(stimulus)):
    for j in range(len(stimulus)):
        if j>i:
            rdm_tri_vercor_01.append(tri01[i][j])
#pccs = pearsonr(x, y)

#绘制相关系数的热力图
import seaborn as sb
#r_pearson = boston.corr()
sb.heatmap(data = rdm_00[7],cmap="RdBu_r")#cmap设置色系
'''