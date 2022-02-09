# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:27:19 2022

@author: Antony
"""

# mlp for the blobs problem with minibatch gradient descent with varied batch size

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2

# fit a model and plot learning curve
def prepare_data(dataset):
    """
    Préparation des sets d'entraînements et de test
    """
    data_train, data_test = train_test_split(dataset, test_size=0.2,  shuffle=True) #useful function
    trainy,trainX=data_train['time'],data_train.drop(['time'],axis=1)
    testy,testX= np.asarray(data_test['time']),np.asarray(data_test.drop(['time'],axis=1))
    # prepare multi-class classification dataset.astype(np.float32)
    trainy,testy=np.asarray(to_categorical(trainy)),np.asarray(to_categorical(testy))
    return trainX, trainy, testX, testy
def fit_model(trainX, trainy, testX, testy, n_batch,optimizer,activity_regularizer,n_layers):
    """
    Définit le modèle en fonction de paramètres d'entrée: régularisateur et nombre de couches
    """
	# define model
    n_input, n_classes = trainX.shape[1], testy.shape[1]
    model = Sequential()
    model.add(Dense(120, input_dim=n_input, activation='linear',activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    for _ in range(1, n_layers):
     
        model.add(Dense(100, activation='linear', kernel_initializer='he_uniform',activity_regularizer=activity_regularizer))
        model.add(Activation('relu'))
        model.add(Dense(n_classes, activation='softmax'))
 # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	# fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0, batch_size=n_batch)
	# plot learning curves
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    plt.plot(acc, label='train')
    plt.plot(val_acc, label='test')
    plt.title('batch='+str(n_batch), pad=-40)
    return acc,val_acc
 
# prepare dataset

# create learning curves for different batch sizes
def overview_batch(dataset,optimizer='adam',activity_regularizer=l2(0.001),n_layers=3):
    """
    Main function
    """
    trainX, trainy, testX, testy=prepare_data(dataset)
    batch_sizes = [1,2,4,8,16,32,64,len(trainX)] #taille de 8

    print("Entraînement des modèles en cours...\n ")
    overview=pd.DataFrame(columns={'Layers','Optimiseur','Regularizer','Batch','Train_accuracy','Test_accuracy'})

    acc_list, val_acc_list =  list(), list()
    for i in tqdm(range(len(batch_sizes))):
    	# determine the plot number
       	plot_no = 420 + (i+1)
       	plt.subplot(plot_no)
    	# fit model and plot learning curves for a batch size
        acc,val_acc =  fit_model(trainX, trainy, testX, testy, batch_sizes[i],optimizer,activity_regularizer,n_layers)
        acc_list.append(acc)
        val_acc_list.append(val_acc)
        current_network = {'Layers': n_layers, 'Optimiseur': optimizer, 'Regulariseur':activity_regularizer._keras_api_names_v1[0],'Batch':i,'Train_accuracy':acc_list[-1][-1], 'Test_accuracy': val_acc_list[-1][-1]}
        overview=overview.append(current_network,ignore_index=True)
    # show learning curves
    print("En fonction de ces courbes, il se peut que certains batch fassent  défault, autant ne pas les inclure dans l'étape suivante \n")
    overview.to_excel("Excel/overview_batch.xlsx")
    print("Tableur excel enregistré sous le nom de overview_regularizer.xlsx")
    
    plt.show()