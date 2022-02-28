# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:24:54 2022

@author: Antony

Pour optimiser le nombre de couche et l'optimisateur à l'aide de GridSearchCv

ENTREE: Les couches et optimisateurs 
SORTIE: Les meilleurs modèles conçernant ces paramètres. 
"""

# Use scikit-learn to grid search the number of neurons
import numpy as np
import warnings
import seaborn as sns
import config #variables
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from pactools.grid_search import GridSearchCVProgressBar
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
# Function to create model, required for KerasClassifier

def create_model(layers,activation,optimizer):
    """
    Crée les modèles en fonction de leur paramètres : Nombre de couche et optimiseur  , la fonction d'activation est imposée

    """
    trainX=config.trainX

    n_input, n_classes = trainX.shape[1], config.n_classes 

	# create model
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=n_input, activation=activation, kernel_initializer='he_uniform'))
        else:
            model.add(Dense(nodes, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))

	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def tuning_layers(layers=[[120],[120,100]],optimizer = ['RMSprop', 'Adam', 'Adamax', 'Nadam'],activation=['relu'] ):
    """
    Implémente le calcul en parallèle de chaque réseau afin d'en selectionner les 5 premiers
    """
    
    trainX,trainy=config.trainX,config.trainy # Récupération du set d'entrée et de sortie
    model = KerasClassifier(build_fn=create_model, epochs=300, verbose=0)
    cv = KFold(n_splits=5,shuffle=True) #Cette cross-validation est obligatoire pour le t-test
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)  #Attribut qui arrête d'entraîner un modèle dès que celui-ci stagne sur ces performances sur le set test
    
    
    ###Définition de la grille de recherche###

    param_grid = dict(activation=activation,layers=layers,optimizer=optimizer) #building the dictionnary to grid
    print()
    print("Nos réseaux seront : \n" ,param_grid)
    grid = GridSearchCVProgressBar(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv)
    print()
    grid.__bar__()
    grid_result = grid.fit(trainX, trainy,callbacks=[callback])


    ###Résumé des résultats###
    print()
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print()
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    
    ###On garde les 10 premiers###
    
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    if len(results[results['mean_test_score']>0.60])!=0:
        results_2=results[results['mean_test_score']>0.60][:10].copy()
    else:
        results_2=results[:5].copy()
    
    
    ###On construit la heatmap en fonction du nombre de couche ###
    
    results=results[results['param_activation']==grid_result.best_params_['activation']]
    n_couches=[len(results['param_layers'].iloc[i]) for i in range(0,len(results))]
    number_layer=pd.DataFrame(n_couches,index=results.index,columns=['number layer'])
    results=pd.concat([results,number_layer],axis=1)
    #Création du pivot et affichage de la heat map#
    pvt = pd.pivot_table(results,
        values='mean_test_score', index='number layer', columns='param_optimizer')
    plt.clf()
    ax = sns.heatmap(pvt,annot=True )
    ax.set_title("Heatmap pour des réseaux avec une couche d'entrée de %s" %layers[0][0])
    fig=ax.get_figure()
    return results_2,fig
