# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 15:24:54 2022

@author: Antony
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
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Function to create model, required for KerasClassifier

def create_model(layers,activation,optimizer):
    trainX,trainy=config.trainX,config.trainy

    n_input, n_classes = trainX.shape[1], config.n_classes #Attention ! 

	# create model
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=n_input, activation=activation, kernel_initializer='he_uniform'))
        else:
            model.add(Dense(nodes, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))

	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# create model
# define the grid search parameters$


def tuning_layers(layers=[[120],[120,100]],activation=['relu'],optimizer = ['RMSprop', 'Adam', 'Adamax', 'Nadam'] ):
    trainX,trainy=config.trainX,config.trainy
    print(layers)
    model = KerasClassifier(build_fn=create_model, epochs=10, verbose=1)
    
    print("Nos réseaux seront : " , layers)
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50) #Arrête de s'entraîner dès que le modèle stagne

    param_grid = dict(activation=activation,layers=layers,optimizer=optimizer) #building the dictionnary to grid
    print(param_grid)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(trainX, trainy,callbacks=[callback])
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    ### transforming testy and trainy in great array
    print()
    print("To give an idea \n")
    trainy=list(map(np.argmax,trainy))
    ###classification report
    print(classification_report(trainy,grid_result.best_estimator_.predict(trainX)))
    ###keeping the tenth first
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    results_2=results.iloc[:10]
    ###prepare the heatmap
    results=results[results['param_activation']==grid_result.best_params_['activation']]
    n_couches=[len(results['param_layers'].iloc[i]) for i in range(0,len(results))]
    number_layer=pd.DataFrame(n_couches,index=results.index,columns=['number layer'])
    results=pd.concat([results,number_layer],axis=1)
    pvt = pd.pivot_table(results,
        values='mean_test_score', index='number layer', columns='param_optimizer')
    plt.clf()
    ax = sns.heatmap(pvt,annot=True )
    ax.set_title("heatmap avec la fonction d'activation du meilleur résultat fixé")
    return results_2
