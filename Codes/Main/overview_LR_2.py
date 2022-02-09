"""
Phase de préparamétrage :
    On identifie les pas d'apprentissages les plus cohérents afin d'éviter de faire trop 
    de calculs lors de l'étape suivante qui consistera à rechercher un réseau optimisé.
Sortie : Des graphiques permettant de mieux comprendre les choix du modèle. 
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
import config
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# monitor the learning rate
class LearningRateMonitor(Callback):
    """
    Classe afin de rassembler aisément nos données sur les performances de chaque modèle
    """
    # start of training
    def on_train_begin(self, logs={}):
        self.lrates = list()
   
   	# end of each training epoch
    def on_epoch_end(self, epoch, logs={}):
   		# get and store the learning rate
   		lrate = float(backend.get_value(self.model.optimizer.lr))
   		self.lrates.append(lrate)

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
def fit_model(trainX, trainy, testX, testy, optimizer,n_layers):
	# define model
    """
    Définit le modèle en fonction de paramètres d'entrée: optimiseur et nombre de couches
    """
    n_input, n_classes = trainX.shape[1], config.n_classes
    model = Sequential()
    model.add(Dense(120, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    for _ in range(1, n_layers):
        
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(n_classes, activation='softmax'))
    # compile model
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0,callbacks=[callback])

    return  history.history['loss'], history.history['accuracy'], history.history['val_accuracy']

# create line plots for a series
def line_plots(momentums, series,n_layers,ax):
    """
    Gère l'affichage
    """
    ax.plot(series,label=str(momentums))
    ax.legend()

def overview_LR(dataset,n_layers= 4):
    """
    Main function
    """
    overview=pd.DataFrame(columns={'Layers','Optimizer','Train_accuracy','Test_accuracy','Loss'})
    # prepare dataset
    trainX, trainy, testX, testy = prepare_data(dataset)
    # create learning curves for different patiences
    momentums = ['sgd', 'rmsprop', 'adagrad', 'adam']
    #momentums = ['adam']
    
    fig1, axs1 = plt.subplots(1, 1, figsize=(20,10))
    fig1.suptitle("Loss depending on optimizer with n_layers = %s"%n_layers)
    fig2, axs2 = plt.subplots(1, 1, figsize=(20,10))
    fig2.suptitle("Train accuracy depending on optimizer with n_layers = %s" %n_layers)
    fig3, axs3 = plt.subplots(1, 1, figsize=(20,10))
    fig3.suptitle("Test accuracy depending on optimizer with n_layers = %s" %n_layers)

    
    # plot learning rates
    print("Entraînement des modèles en cours...\n ")
    
    for i in tqdm(range(len(momentums))):
        loss_list, acc_list,val_acc_list =  list(), list(),list()

        loss, acc,val_acc = fit_model(trainX, trainy, testX, testy, momentums[i],n_layers)
        loss_list.append(loss)
        acc_list.append(acc)
        val_acc_list.append(val_acc)
        current_network = {'Layers': n_layers, 'Optimizer': momentums[i], 'Train_accuracy':acc_list[0][-1], 'Test_accuracy': val_acc_list[0][-1],'Loss':loss_list[0][-1]}
        overview=overview.append(current_network,ignore_index=True)
    # plot loss
        line_plots(momentums[i], loss_list[0],n_layers,axs1)
        line_plots(momentums[i], acc_list[0],n_layers,axs2)
        line_plots(momentums[i], val_acc_list[0],n_layers,axs3)
    fig1.savefig("Plots/Pre_parametrage/Loss_LR")
    fig2.savefig("Plots/Pre_parametrage/Train_LR")
    fig3.savefig("Plots/Pre_parametrage/Test_LR")
    overview.sort_values(by='Test_accuracy', inplace=True) #pas de overview=....sort_values !!!
    overview=overview[overview['Train_accuracy']>0.70] #On récupère les modèles potables
    if len(overview)==0:
        print("Pas de modèles concluant , on garde tout les optimisateurs")
        return momentums
            
    overview.to_excel("Excel/overview_optimizer.xlsx")
    optimisateurs=np.unique(list(overview['Optimizer'][:4])) #On garde les 4 premiers optimiseurs
    print("Tableur excel enregistré sous le nom de overview_optimizer.xlsx")
    print("Les optimiseurs conservés sont %s "%optimisateurs)
    return optimisateurs

