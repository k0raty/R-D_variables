# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:19:10 2022

@author: Antony
"""
import pandas as pd
from tensorflow.keras.utils import to_categorical
import numpy as np
file="data_classed.pkl"
data=pd.read_pickle("data_classed.pkl")
data=data[data.columns[:9]] #Ne pas oublier ! 
n_classes=data['time'].drop_duplicates().max()+1
def prepare_data(dataset):


    #####splitting
    trainy,trainX=np.asarray(dataset['time']),np.asarray(dataset.drop(['time'],axis=1))
    # prepare multi-class classification dataset
    trainy=np.asarray(to_categorical(trainy))
    return trainX, trainy
    # fit a model and plot learning curve
def create_layer(n_layer,n_neuron,n_classes=n_classes): #adapter en fonction du nombre de classe
    sec=2*n_classes
    if n_layer>int((np.log(n_neuron)-np.log(sec))/np.log(2)):
        n_layer_2 = int((np.log(n_neuron)-np.log(sec))/np.log(2))
    else:
        n_layer_2=n_layer
    layers=[n_neuron]
    for j in range(0,n_layer_2-1):
        if layers[-1]//2 >sec:
            layers.append(layers[-1]//2)
    list_neurons= [layers[:i] for i in range(1,len(layers)+1)]
    if n_layer_2 != n_layer:
        print("la liste est",list_neurons)
        for i in range(1,n_layer-n_layer_2+1):
            print("i est",i)
            print(list_neurons[-1])
            
            list_neurons.append(list_neurons[-1]+[sec])
    return list_neurons
trainX,trainy=prepare_data(data)
