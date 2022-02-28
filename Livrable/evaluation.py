#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:53:17 2022

@author: Antony

Evalue les performance du modèle sur un set d'entraînement et de test.
Puis les performances sur le set original sans réduction des élèments. 
"""
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import config 
from silence_tensorflow import silence_tensorflow
import warnings 

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'
silence_tensorflow()
def eval_processed_set(model,x_train,y_train,x_test,y_test):
    
    """
    Evaluation du modèle choisi sur le set réarrangé , dans lequel les doublons ont été retiré.
    ENTREE : Modèle et set de données
    SORTIE : Heatmaps
    """
    

    ###Test du modèle###
    xticklabels=np.sort(pd.unique(config.dataset['time']))*10
    predictions = model(x_test)
    prediction = tf.nn.softmax(predictions).numpy()
    y_pred_test = [np.argmax(prediction[i]) for i in range(0, len(prediction))]
    fig, ax1 = plt.subplots(1, 1)
    #Heatmaps et rapport de classification#
    g1 = sns.heatmap(confusion_matrix(y_pred_test, y_test), xticklabels=xticklabels, yticklabels=xticklabels,annot=True, ax=ax1)
    g1.set_ylabel('y_test')
    g1.set_xlabel('y_pred')
    g1.set_title("Heatmap for test set")
    fig.savefig("Evaluation/Heatmap_test_set")
    
    class_report= classification_report(y_test,y_pred_test,output_dict=True)
    df=pd.DataFrame.from_dict(class_report)
    df.to_excel("Evaluation/class_report_test_set.xlsx")
    print("Report for test neural network : \n",
          classification_report(y_test,y_pred_test))
    
    predictions = model(x_train)
    prediction = tf.nn.softmax(predictions).numpy()
    y_pred_train = [np.argmax(prediction[i]) for i in range(0, len(prediction))]
    fig, ax2 = plt.subplots(1, 1)
    
    g2 = sns.heatmap(confusion_matrix(y_pred_train, y_train), xticklabels=xticklabels, yticklabels=xticklabels,annot=True, ax=ax2)
    g2.set_ylabel('y_train')
    g2.set_xlabel('y_pred')
    g2.set_title("Heatmap for training set")
    fig.savefig("Evaluation/heatmap_train_set")
    class_report=classification_report(y_train, y_pred_train,output_dict=True)
    df=pd.DataFrame.from_dict(class_report)
    df.to_excel("Evaluation/class_report_train_set.xlsx")
    print("Report for train neural network : \n",
          classification_report(y_train, y_pred_train))
    
    return y_pred_train,y_pred_test

def isnan(num):
    return num != num

def eval_init_set(y, data_set, data,title,security):
    
    """
    Evaluation du modèle choisi sur le set initial , dans lequel 
    on évalue "les temps des véritables élèments"  et non par rapport à la moyenne .
    
    ENTREE : Modèle et set de données
    SORTIE : Heatmaps
    """
    
    datatrue = data_set.copy()
    data['class_pred']=np.nan    
    data['class_true'] = [round(data['time'].iloc[i]/10) for i in range(len(data['time']))] #On affecte à chaque élèment sa vraie classe
    ###Pour chaque élèment on affecte sa classe prédie par le modèle.###
    for i in range(0, len(datatrue)):
        id_same_vector = datatrue['index'].iloc[i] #Eléments ayant le même vecteur
        data['class_pred'].loc[id_same_vector] = y[i]
        # on redistribue le temps sous forme d'intervalle.
    data=data[isnan(data['class_pred'])==False] #we keep only validate element
    data['sucess']=""
    
    ###On vérifie la précision du modèle lorsque qu'on affecte chaque élèments à son intervalle de prédiction###
    print("Affectation des élèments à sa classe d'intervalle en cours ....\n")
    for i in tqdm(range(0, len(data))):
        truth = (data['class_pred'].iloc[i]*10-security) < data.iloc[i]['time'] < (data['class_pred'].iloc[i]*10+security)
        data["sucess"].iloc[i] = int(truth)   
    ratio=len(data[data["sucess"]==1])/len(data)
    print("Le ratio d'exactitute est : \n", ratio)
    
    ###Heatmaps et rapport de classification###
    fig, ax3= plt.subplots(1, 1)
    xticklabels=np.sort(pd.unique(data['class_true']))*10 #Pour rétablir en légende les classes correctes. 
    
    g3 = sns.heatmap(confusion_matrix(data['class_true'],data['class_pred']),xticklabels=xticklabels, yticklabels=xticklabels,annot=True, ax=ax3)
    g3.set_ylabel('true (en ms)')
    g3.set_xlabel('pred (en ms)')
    g3.set_title("Real confusion matrix on %s +/- %s ms avec un ratio de %.3f " %(title,security,ratio))
    fig.savefig("Evaluation/Real_%s" %title)
    class_report=classification_report(
        data['class_true'],data['class_pred'],output_dict=True)
    df=pd.DataFrame.from_dict(class_report)
    df.to_excel("Evaluation/Confusion_matrix_Real_%s.xlsx" %title)
    print("Rapport approché de %s : \n" %title , classification_report(
        data['class_true'],data['class_pred']))
def evaluation(model,security,data_train,data_test,x_train,y_train,x_test,y_test):
    """
    Main function 
    """
    data=config.data
    y_pred_train,y_pred_test=eval_processed_set(model,x_train,y_train,x_test,y_test) 
    eval_init_set(y_pred_train, data_train,data ,"training set",security)
    eval_init_set(y_pred_test, data_test,data, "test set",security)

