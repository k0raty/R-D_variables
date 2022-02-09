"""
Phase de préparamétrage :
    On identifie les régularisateurs les plus efficaces afin d'éviter de faire trop 
    de calculs lors de l'étape suivante qui consistera à rechercher un réseau optimisé.
Sortie : Des graphiques nécessitant une analyse humaine afin d'établir des conclusions cohérente
C'est une étape optionnelle
"""
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
    n_input, n_classes = trainX.shape[1], testy.shape[1]
    model = Sequential()
    model.add(Dense(120, input_dim=n_input, activation='linear',activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    for _ in range(1, n_layers):
        
        model.add(Dense(100, activation='linear', kernel_initializer='he_uniform',activity_regularizer=activity_regularizer))
        model.add(Activation('relu'))
    model.add(Dense(n_classes, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)

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

def overview_regularizer(dataset,optimizer='adam',num_layers=np.array([0,1,2,3,4])):
    regularizers = [None,l1(0.001),l2(0.001),l1_l2(0.001)]
    fig1, axs1 = plt.subplots(1, len(regularizers), figsize=(20,10))
    fig1.suptitle("Loss depending on optimizer")
    fig2, axs2 = plt.subplots(1, len(regularizers), figsize=(20,10))
    fig2.suptitle("Accuracy depending on optimizer")
    fig3, axs3 = plt.subplots(1, len(regularizers), figsize=(20,10))
    fig3.suptitle('Generalization error')
    # plot learning rates
    overview=pd.DataFrame(columns={'Layers','Regularizer','Train_accuracy','Test_accuracy','Loss'})
    # prepare dataset
    trainX, trainy, testX, testy = prepare_data(dataset)
    print("Entraînement des modèles en cours...\n ")
    for n_layers in tqdm(num_layers):
        loss_list, acc_list, val_acc_list =  list(), list(),list()
    
        for i in range(len(regularizers)):
            loss, acc ,val_acc= fit_model(trainX, trainy, testX, testy, regularizers[i],n_layers,optimizer)
            loss_list.append(loss)
            acc_list.append(acc)
            val_acc_list.append(val_acc)
            print()
            if None in regularizers: #Check for None value , when we do not want to use a regularizer 
                index=regularizers.index(None)
            else: index=-1
            if i==index:
                current_network = {'Layers': n_layers, 'Optimiseur':optimizer, 'Regularizer': 'Aucun', 'Train_accuracy':acc_list[-1][-1], 'Test_accuracy': val_acc_list[-1][-1],'Loss':loss_list[-1][-1]}

            else : 
                current_network = {'Layers': n_layers,'Optimiseur':optimizer, 'Regularizer': regularizers[i]._keras_api_names_v1[0], 'Train_accuracy':acc_list[-1][-1], 'Test_accuracy': val_acc_list[-1][-1],'Loss':loss_list[-1][-1]}
            overview=overview.append(current_network,ignore_index=True)
        # plot loss
        if len(regularizers)==1:
            line_plots(regularizers[0], loss_list[0],n_layers,axs1)
            line_plots(regularizers[0], acc_list[0],n_layers,axs2)
            line_plots(regularizers[0], val_acc_list[0],n_layers,axs3)
    
        else:
            for l, ax in enumerate(axs1.flat):
        
                line_plots(regularizers[l], loss_list[l],n_layers,ax)
            # plot accuracy
            for l, ax in enumerate(axs2.flat):
        
                line_plots(regularizers[l], acc_list[l],n_layers,ax)
            for l, ax in enumerate(axs3.flat):
        
                line_plots(regularizers[l], val_acc_list[l],n_layers,ax)
    print("En fonction de ces courbes, il se peut que certains régularisateurs fassent  défault, autant ne pas les inclure dans l'étape suivante \n")
    overview.to_excel("Excel/overview_regularizer.xlsx")
    print("Tableur excel enregistré sous le nom de overview_regularizer.xlsx")
    
