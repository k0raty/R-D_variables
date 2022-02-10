# -*- coding: utf-8 -*-
"""
Fonction pour optimiser le regularisateur à l'aide de GridSearchCv
"""
import config
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import seaborn as sns
from pactools.grid_search import GridSearchCVProgressBar
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2

def create_model(parameters,activity_regularizer):
    """
    Crée les modèles en fonction de leur paramètres de régularisation 

    """
    trainX=config.trainX

    n_input, n_classes = trainX.shape[1], config.n_classes 
    layers,activation,optimizer=parameters['layers'],parameters['activation'],parameters['optimizer']
	# create model
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=n_input, activation=activation,activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
        else:
            model.add(Dense(nodes, activation=activation,activity_regularizer=activity_regularizer))
    model.add(Dense(n_classes, activation='softmax'))

	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model




def tuning_regularizer(results,coefficient,activity_regularizer = [None,l1(0.001),l2(0.001)]):
    """
    Implémente le calcul en parallèle de chaque réseau afin d'en selectionner les 5 premiers
    """
    
    trainX,trainy=config.trainX,config.trainy #Récuperation des set d'entrée et de sortie 
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50) #Attribut qui arrête d'entraîner un modèle dès que celui-ci stagne sur ces performances sur le set test

    model = KerasClassifier(build_fn=create_model, epochs=5, verbose=0) #Définition du modèle 
    parameters= dict(results['params'])
    print("Nos réseaux seront : " , parameters)
    parameters=[parameters[i] for i in parameters.keys()]
    
    param_grid = dict(parameters=parameters, activity_regularizer=activity_regularizer)  #On récupère les paramètres des modèles sauvegardés et on y ajoute un régularisateur
    
    
    ###Définition de la grille de recherche###
    
    grid = GridSearchCVProgressBar(estimator=model, param_grid=param_grid, n_jobs=-2, cv=3) 
    grid.__bar__()
    grid_result = grid.fit(trainX, trainy,callbacks=[callback]) 
    
    ###Résumé des résultats###
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
   
    
    ###On gardes les 5 premiers###
    
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    results_2=results.iloc[:5] 
    
    
    ###Affichage de la Heatmap###
    
    slicing=[results['param_parameters'].iloc[i]['optimizer']==grid_result.best_params_['parameters']['optimizer'] for i in range(0,len(results))] #On ne garde que les élèment ayant le même optimiseur que le meilleur résultat 
    results=results[slicing]
    results['regularizer']="" #On crée une colonne qui regroupe les vrais nom des régularisateurs et non leur nom d'objet
    for i in range(0,len(results)):
        regularisateur= results['param_activity_regularizer'].iloc[i]
        if regularisateur != None:
            results['regularizer'].iloc[i]=regularisateur._keras_api_names_v1[0]
        else :
            results['regularizer'].iloc[i] = 'None'
    n_couches=[len(results['param_parameters'].iloc[i]['layers']) for i in range(0,len(results))] #On identifie le nombre de couche de chaque réseau pour la heatmap
    number_layer=pd.DataFrame(n_couches,index=results.index,columns=['number layer'])
    results=pd.concat([results,number_layer],axis=1)
    
    #Création du pivot et affichage de la heat map#
    pvt = pd.pivot_table(results,
        values='mean_test_score', index='number layer', columns='regularizer') 
    plt.clf()
    ax = sns.heatmap(pvt,annot=True )
    ax.set_title("Heatmaps des réseaux avec l'optimisateur du meilleur résultat :%s et un coefficient de régularisation : %s "%(grid_result.best_params_['parameters']['optimizer'],coefficient))
    return results_2
