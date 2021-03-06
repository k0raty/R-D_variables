# -*- coding: utf-8 -*-
"""
Fonction pour optimiser le regularisateur à l'aide de GridSearchCv.

ENTREE: Les meilleurs modèles isssues de tuning_layer.py et les régularisateurs conseillés
SORTIE: Les meilleurs modèles conçernant ces paramètres. 
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
from sklearn.model_selection import KFold 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
pd.options.mode.chained_assignment = None  # default='warn'

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
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model




def tuning_regularizer(results,coefficient,activity_regularizer = [None,l1(0.001),l2(0.001)]):
    """
    Implémente le calcul en parallèle de chaque réseau afin d'en selectionner les 5 premiers
    """
    
    trainX,trainy=config.trainX,config.trainy #Récuperation des set d'entrée et de sortie 
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50) #Attribut qui arrête d'entraîner un modèle dès que celui-ci stagne sur ces performances sur le set test
    cv = KFold(n_splits=5,shuffle=True) #Cette cross-validation est obligatoire pour le t-test
    model = KerasClassifier(build_fn=create_model, epochs=300, verbose=0) #Définition du modèle 
    parameters= dict(results['params'])
    parameters=[parameters[i] for i in parameters.keys()]
    
    param_grid = dict(parameters=parameters, activity_regularizer=activity_regularizer)  #On récupère les paramètres des modèles sauvegardés et on y ajoute un régularisateur
    print()
    print("Nos réseaux seront : \n" ,param_grid)

    ###Définition de la grille de recherche###
    
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
   
    
    ###On gardes les plus intéressants###
    
    results = pd.DataFrame(grid.cv_results_)
    results.sort_values(by='rank_test_score', inplace=True)
    if len(results[results['mean_test_score']>0.70])!=0:
        results_2=results[results['mean_test_score']>0.70][:10].copy()
    else:
        results_2=results[:5]  .copy()  
    
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
    fig=ax.get_figure()
    return results_2,fig
