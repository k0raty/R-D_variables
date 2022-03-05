# -*- coding: utf-8 -*-
"""
Fichier contenant les constantes importantes à globaliser
"""
import pandas as pd
from tensorflow.keras.utils import to_categorical
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
data=pd.read_pickle("DataFrame/Pré-processing/data_vectorized.pkl")
dataset_stats=pd.read_pickle("DataFrame/Pré-processing/data_classed.pkl")
dataset = dataset_stats[dataset_stats.columns[:len(data.columns)]] #C'est le set final sans les informations statistiques
n_classes=int(dataset['time'].drop_duplicates().max()+1)
def prepare_data(dataset):


    #####splitting
    trainy_1D,trainX=np.asarray(dataset['time']),np.asarray(dataset.drop(['time'],axis=1))
    # prepare multi-class classification dataset
    trainy=np.asarray(to_categorical(trainy_1D,n_classes))
    return trainX, trainy,trainy_1D
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
        for i in range(1,n_layer-n_layer_2+1):
            list_neurons.append((list_neurons[-1]+[sec]))
    return list_neurons
trainX,trainy,trainy_1D=prepare_data(dataset)

