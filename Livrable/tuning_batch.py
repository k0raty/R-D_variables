# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:18:22 2022

@author: Antony

On optimise le batch à l'aide de GridSearchCv , critère important : 

https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/
    

ENTREE: Les meilleurs modèles choisis par tuning_regularizer.py ainsi que le batch minimum
SORTIE: Les meilleurs modèles conçernant ces paramètres. 

"""
import config 
import tensorflow as tf
from pactools.grid_search import GridSearchCVProgressBar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

def create_model(parameters):
    """
    Crée les modèles en fonction de leur paramètres : Nombre de couche et optimiseur et regularisateur, la fonction d'activation est imposée

    """
    trainX=config.trainX

    n_input, n_classes = trainX.shape[1], config.n_classes

    layers,activation,optimizer,activity_regularizer=parameters['parameters']['layers'],parameters['parameters']['activation'],parameters['parameters']['optimizer'],parameters['activity_regularizer']
	# create model
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=n_input, activation=activation,activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
        else:
            model.add(Dense(nodes, activation=activation,activity_regularizer=activity_regularizer))
    model.add(Dense(n_classes, activation='softmax'))

	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
def create_batch(n):
    """
    Crée la liste des batchs à évaluer selon une logique reconnue (puissance de 2)
    """
    batch=[]
    i=0
    while pow(2,i) < n:
        batch.append(pow(2,i))
        i+=1
    batch.append(n)
    return batch
# define the grid search parameters$
def tuning_batch(results,batch, epochs=300):
    """
    Implémente le calcul en parallèle de chaque réseau afin d'en selectionner les 5 premiers
    """
    trainX,trainy=config.trainX,config.trainy # Récupération du set d'entrée et de sortie

    results=results[:4]
    model = KerasClassifier(build_fn=create_model, verbose=0)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50) #Attribut qui arrête d'entraîner un modèle dès que celui-ci stagne sur ces performances sur le set test
    
    
    ###Définition de la grille de recherche###
    
    parameters= dict(results['params'])
    param=[parameters[i] for i in parameters.keys()]#On récupère les paramètres précédents
    #Liste_batch#
    batchsize= create_batch(len(trainX)) #On crée la liste des batchs
    index=batchsize.index(batch)
    batchsize=batchsize[index:] #On écarte les batchs trop gourmand en ressource
    if len(batchsize)>5:
        batchsize=batchsize[:5] #Plus de 10 batchs n'est pas utile
    batchsize.append(None)
    epochs=[epochs]
    param_grid = dict(parameters=param,batch_size=batchsize,epochs=epochs)
    print()
    print("Nos réseaux seront : \n" ,param_grid)
    
    cv = KFold(n_splits=5,shuffle=True) #Cette cross-validation est obligatoire pour le t-test

    grid = GridSearchCVProgressBar(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv)
    print()
    grid.__bar__()
    grid_result = grid.fit(trainX, trainy,callbacks=[callback])
    
    ### Résumé des résultats ###
    print()
    print("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    print()
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    best_results=results[:5]
    if len(results[results['mean_test_score']>0.75])!=0:
        results_2=results[results['mean_test_score']>0.75][:5].copy()
    else:
        results_2=results[:3].copy()
    
    ###Affichage des 5 premiers résultats###
    
    dataplotting=results.copy()
    dejavu=[]
    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    for i in range(0,len(best_results)):
        if best_results['param_parameters'].iloc[i] not in dejavu: #imaginons que on est les même paramètres mais pas le même nombre d'epoch dans les meilleurs résultats
            
            batch=list(dataplotting['param_batch_size'][dataplotting['param_parameters']==best_results['param_parameters'].iloc[i]])
            if None in batch:
                index=batch.index(None)
                batch[index]=0
            std_dev= dataplotting['std_test_score'][dataplotting['param_parameters']==best_results['param_parameters'].iloc[i]]
            accuracy= dataplotting['mean_test_score'][dataplotting['param_parameters']==best_results['param_parameters'].iloc[i]]
            ax.errorbar(batch, accuracy, std_dev, linestyle='None', marker='o',label=best_results['rank_test_score'].iloc[i])
            ax.legend()
            ax.set_title(" Evolution des résultats au test : Cross-validation pour au plus les 5 premiers modèles")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Précision")
            dejavu.append(best_results['param_parameters'].iloc[i])
            fig.savefig("Plots/Paramétrage/Info_batch")
    n_train = len(list(cv.split(trainX, trainy))[0][0])
    n_test = len(list(cv.split(trainX, trainy))[0][1])

    return results_2, n_train, n_test
