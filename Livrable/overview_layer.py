"""
Phase de préparamétrage :
    On identifie le nombre de couches par neurone le plus efficaces afin d'éviter de faire trop
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
import warnings
from silence_tensorflow import silence_tensorflow
import config
silence_tensorflow()
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
    trainy,testy=np.asarray(to_categorical(trainy,config.n_classes)),np.asarray(to_categorical(testy,config.n_classes))
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
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50) #Arrête de s'entraîner dès que le modèle stagne

    model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0,callbacks=[callback])

    return  history.history['loss'], history.history['accuracy'],history.history['val_accuracy']

# create line plots for a series
def line_plots(regularizers, series,n_layers,ax):
    """
    Gère l'affichage
    """
    ax.plot(series,label=str(n_layers))
    ax.legend()
    if regularizers==None:
        ax.set_title('regularizer= None', pad=-80)
    else :
        ax.set_title('regularizer='+regularizers._keras_api_names_v1[0], pad=-80)


def overview_layer(dataset,num_layers=np.array([1,2,3,4,5,6,7,8,9,10]),best_score=float('inf'),optimizer='adam',regularizer=None):
    fig1, axs1 = plt.subplots(1, 1, figsize=(20,10))
    fig1.suptitle("Loss depending on layers")
    fig2, axs2 = plt.subplots(1, 1, figsize=(20,10))
    fig2.suptitle("Accuracy depending on layers")
    fig3, axs3 = plt.subplots(1, 1, figsize=(20,10))
    fig3.suptitle('Generalization error')
    # plot learning rates
    overview=pd.DataFrame(columns={'Layers','Regularizer','Train_accuracy','Test_accuracy','Loss'})
    # prepare dataset
    trainX, trainy, testX, testy = prepare_data(dataset)
    print("Entraînement des modèles en cours...\n ")
    for n_layers in tqdm(num_layers):
        loss_list, acc_list, val_acc_list =  list(), list(),list()
        loss, acc ,val_acc= fit_model(trainX, trainy, testX, testy, regularizer,n_layers,optimizer)
        loss_list.append(loss)
        acc_list.append(acc)
        val_acc_list.append(val_acc)
        current_network = {'Layers': n_layers,'Optimiseur':optimizer, 'Regularizer': regularizer, 'Train_accuracy':acc_list[0][-1], 'Test_accuracy': val_acc_list[0][-1],'Loss':loss_list[0][-1]}
        overview=overview.append(current_network,ignore_index=True)
        # plot lines
        line_plots(regularizer, loss_list[0],n_layers,axs1)
        line_plots(regularizer, acc_list[0],n_layers,axs2)
        line_plots(regularizer, val_acc_list[0],n_layers,axs3)

    overview.sort_values(by='Train_accuracy')
    current_best_score=overview.iloc[0]['Train_accuracy']
    print("Le meilleur score est : \n", current_best_score)
    overview=overview[overview['Train_accuracy']>0.65] #On récupère les modèles potables

    if len(overview)==0:
        if current_best_score<best_score : 
            num_layers=num_layers+10*np.ones(len(num_layers))
            num_layers=num_layers.astype(int)
            print("Pas de modèles concluant , tentative avec un nombre de couche plus élevé : %s \n" %num_layers)
            return overview_layer(dataset,num_layers,current_best_score)
        else: 
            print("Malheureusement ,pas de modèle concluant trouvé dans l'absolu \n")
            
    overview.sort_values(by='Test_accuracy', inplace=True)
    layers=min(np.unique(list(overview['Layers'][:5]))) #On garde le minimum des 5 premières couches
    layers=np.arange(layers,layers+5,1)
    fig1.savefig("Plots/Pré-paramétrage/Loss_layer")
    fig2.savefig("Plots/Pré-paramétrage/Train_layer")
    fig3.savefig("Plots/Pré-paramétrage/Test_layer")
    overview.to_excel("Excel/Pré-paramétrage/overview_num_layers.xlsx")
    overview.to_pickle("DataFrame/Pré-paramétrage/overview_num_layers.pkl")

    print("Tableurs enregistrés sous le nom de overview_num_layers")
    print("Les nombres de couches qui seront conservés sont %s "%layers)
    return layers

