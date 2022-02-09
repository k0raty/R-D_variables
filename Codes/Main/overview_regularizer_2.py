"""
Phase de préparamétrage :
    On identifie les régularisateurs les plus efficaces afin d'éviter de faire trop 
    de calculs lors de l'étape suivante qui consistera à rechercher un réseau optimisé.
Sortie : Des graphiques permettant de mieux comprendre les choix du modèle. 
C'est une étape optionnelle
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #Ignore des Warnings inutiles
# prepare train and test dataset
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
# fit a model and plot learning curve
def fit_model(trainX, trainy, testX, testy, activity_regularizer,n_layers,optimizer):
	# define model
    """
    Définit le modèle en fonction de paramètres d'entrée: régularisateur et nombre de couches
    """
    n_input, n_classes = trainX.shape[1], config.n_classes
    model = Sequential()
    model.add(Dense(120, input_dim=n_input, activation='linear',activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    for _ in range(1, n_layers):
        
        model.add(Dense(100, activation='linear', kernel_initializer='he_uniform',activity_regularizer=activity_regularizer))
        model.add(Activation('relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # compile model
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50) #Arrête de s'entraîner dès que le modèle stagne

    model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0,callbacks=[callback])

    return  history.history['loss'], history.history['accuracy'],history.history['val_accuracy']

# create line plots for a series
def line_plots(regularizers, series,n_layers,ax):
    """
    Gère l'affichage
    """
    if regularizers==None:
        ax.plot(series,label='None')

    else :
        ax.plot(series,label=regularizers._keras_api_names_v1[0])
    ax.legend()

def overview_regularizer(dataset,optimizer='adam',n_layers=4):
    """
    Main function
    """

    regularizers = [None,l1(0.001),l2(0.001),l1_l2(0.001)]
    fig1, axs1 = plt.subplots(1, 1, figsize=(20,10))
    fig1.suptitle("Loss depending on regularizer with n_layers = %s"%n_layers)
    fig2, axs2 = plt.subplots(1, 1, figsize=(20,10))
    fig2.suptitle("Accuracy depending on regularizer with n_layers = %s"%n_layers)
    fig3, axs3 = plt.subplots(1, 1, figsize=(20,10))
    fig3.suptitle("Generalization error with n_layers = %s" %n_layers)
    # plot learning rates
    overview=pd.DataFrame(columns={'Layers','Regularizer','Train_accuracy','Test_accuracy','Loss'})
    # prepare dataset
    trainX, trainy, testX, testy = prepare_data(dataset)
    print("Entraînement des modèles en cours...\n ")
        
    for i in tqdm(range(len(regularizers))):
        loss_list, acc_list, val_acc_list =  list(), list(),list()
    
        loss, acc ,val_acc= fit_model(trainX, trainy, testX, testy, regularizers[i],n_layers,optimizer)
        loss_list.append(loss)
        acc_list.append(acc)
        val_acc_list.append(val_acc)
        print()
        if None in regularizers: #Check for None value , when we do not want to use a regularizer 
            index=regularizers.index(None)
        else: index=-1
        if i==index:
            current_network = {'Layers': n_layers, 'Optimiseur':optimizer, 'Regularizer': 'Aucun', 'Train_accuracy':acc_list[0][-1], 'Test_accuracy': val_acc_list[0][-1],'Loss':loss_list[0][-1]}

        else : 
            current_network = {'Layers': n_layers,'Optimiseur':optimizer, 'Regularizer': regularizers[i], 'Train_accuracy':acc_list[0][-1], 'Test_accuracy': val_acc_list[0][-1],'Loss':loss_list[0][-1]}
        overview=overview.append(current_network,ignore_index=True)
    # plot loss

        line_plots(regularizers[i], loss_list[0],n_layers,axs1)
        line_plots(regularizers[i], acc_list[0],n_layers,axs2)
        line_plots(regularizers[i], val_acc_list[0],n_layers,axs3)

    fig1.savefig("Plots/Pre_parametrage/Loss_regularizer")
    fig2.savefig("Plots/Pre_parametrage/Train_regularizer")
    fig3.savefig("Plots/Pre_parametrage/Test_regularizer")
    overview.sort_values(by='Test_accuracy', inplace=True)
    overview=overview[overview['Train_accuracy']>0.70] #On récupère les modèles potables
    overview=overview[overview['Test_accuracy']>0.60]
    if len(overview)==0:
        print("Pas de modèles concluant , on garde tout les régularisateurs")
        return regularizers
    regularisateurs=np.unique(list(overview['Regularizer'][:5])) #On garde les 5 premiers optimiseurs
    overview.to_excel("Excel/Pre_parametrage/overview_regularizer.xlsx")
    print("Tableur excel enregistré sous le nom de overview_regularizer.xlsx")
    print("Les regularisateurs conservés sont %s " %regularisateurs)

    return regularisateurs
