#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:01:40 2022

@author: k0raty

Etape effectuée après la comparaison statistique, on compare les modèles performant
de la même manière. L'algorithme compare les performance en temps de réponse des différents
modèles conservés. Allant jusqu'à 1 million d'entrée , il gardera le modèle le plus rapide.
On affichera les performances en temps des différents modèles.  
"""
# -*- coding: utf-8 -*-

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from evaluation import evaluation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import time 
import config 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
term_size = os.get_terminal_size()

def prepare_data(dataset):
    """
    Préparation des sets d'entraînements et de test
    """
    data_train, data_test = train_test_split(
         config.dataset_stats, test_size=0.1,  shuffle=True)  # useful function
    y_train, x_train = data_train['time'].to_numpy(
    ), data_train[data_train.columns[1:len(config.data.columns)]].to_numpy()
    y_test, x_test = data_test['time'].to_numpy(
    ), data_test[data_test.columns[1:len(config.data.columns)]].to_numpy()


    # prepare multi-class classification dataset.astype(np.float32)
    trainy,testy=np.asarray(to_categorical(y_train,config.n_classes)),np.asarray(to_categorical(y_test,config.n_classes))
    return x_train, y_train, x_test,y_test, trainy, testy,data_train,data_test

# prepare train and test dataset
def create_model(parameters,trainX):
    """
    Crée les modèles en fonction de leurs paramètres de régularisation 

    """
    n_input, n_classes = trainX.shape[1], config.n_classes 
    layers,activity_regularizer,optimizer=parameters['layers'],parameters['activity_regularizer'],parameters['optimizer']
	# create model
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=n_input, activation='relu',activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
        else:
            model.add(Dense(nodes, activation='relu',activity_regularizer=activity_regularizer))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
    # fit model
def fit_model(model,parameters,trainX, trainy, testX, testy):
    """
    Fit the model
    
    """
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50)  #Attribut qui arrête d'entraîner un modèle dès que celui-ci stagne sur ces performances sur le set test

    history = model.fit(trainX, trainy, validation_data=(testX, testy),epochs=300, verbose=0,batch_size=parameters['batch_size'],callbacks=[callback])

    return  history.history['loss'], history.history['accuracy']


def tuning_time(results,security):
    """
    Main function 
    """
    plt.clf()
    dataset=config.dataset
    x_train, y_train, x_test,y_test, trainy, testy,data_train,data_test=prepare_data(dataset)
    elements= [results.iloc[i]['params'] for i in range(0,len(results))]
    parameters=[{'batch_size':i['batch_size'],'activity_regularizer':i['parameters']['activity_regularizer'],'layers':i['parameters']['parameters']['layers'],'optimizer':i['parameters']['parameters']['optimizer']}for i in elements]
    print()
    print("Les modèles comparés sont donc : \n",parameters)
    for i in tqdm(range(0,len(results))):
        params= parameters[i]
        model=create_model(params,x_train)
        fit_model(model,params,x_train, trainy, x_test, testy)
        temps=[]
        best_time= float('inf')
        dataset_2=dataset.drop(['time'],axis=1).copy()
        size_set=[len(dataset_2)]
        while size_set[-1]<200000:
            previous_time=time.time()  
            x_temps=dataset_2.to_numpy()
            predictions = model(x_temps)
            prediction = tf.nn.softmax(predictions).numpy()
            y_pred_test = [np.argmax(prediction[l]) for l in range(0, len(prediction))]
            current_time=time.time()
            time_laps=current_time-previous_time                
            temps.append(float(time_laps))
            dataset_2=pd.concat([dataset_2,dataset_2])
            size_set.append(len(dataset_2))
        size_set.pop() #On retire la dernière taille qui n'aura pas été évalué
        if time_laps <best_time:
            best_time=time_laps
            index=i
            model.save("Evaluation/my_h5_model.h5")
        plt.plot(size_set,temps,'-o',label= "Rang " + str(results.iloc[i]['rank_test_score']))
        plt.legend()
    plt.title("Time_laps depending onlayers")
    plt.xlabel("nombre d'évaluations")
    plt.ylabel("seconde" )
    plt.savefig("Plots/Post_paramétrage/comparaison_des_temps")
    print()
    print("Graph enregistré sous le nom de comparaison_des_temps \n")
    
    ### Sauvegarde du modèle ###
    # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
    
    # It can be used to reconstruct the model identically.
    reconstructed_model = tf.keras.models.load_model("Evaluation/my_h5_model.h5")
    results.iloc[index].to_pickle("Evaluation/chosen_model.pkl")
    results.iloc[index].to_excel("Evaluation/chosen_model.xlsx")

    # Let's check:
    np.testing.assert_allclose(
        model.predict(x_test), reconstructed_model.predict(x_test)
    )
    print("On garde le modèle de rang %s \n" %results.iloc[index]['rank_test_score'])
    print("Ses paramètres sont enregistrés dans : chosen_model.xlsx et chosen_model.pkl dans le dossier Evaluation")
    print("Le modèle est enregistré sous my_h5_model.h5 \n")
    
    ### Evaluation du modèle ###
    
    print('#' * term_size.columns)
    print("Début de la onzième et dernière étape : Evaluation \n")
    print("Ceci est la phase d'évaluation et son dossier respectif à été crée \n")
    return(evaluation(reconstructed_model,security,data_train,data_test,x_train,y_train,x_test,y_test))
