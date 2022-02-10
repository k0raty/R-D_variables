# -*- coding: utf-8 -*-
"""
Programme permettant d'optimiser les différents paramètres du réseau optimal sur plusieurs étape :
    - Optimisation du nombre de couche, de neurones par couches ainsi que de l'optimiseur à prendre en compte
    -Optimisation du régularisateur 
    -Choix parmis les meilleurs modèles

"""


import matplotlib.pyplot as plt 
import os
import shutil
import config
from config import create_layer
from tuning_layers import tuning_layers
from tuning_regularizer import tuning_regularizer
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
term_size = os.get_terminal_size()
def tuning(dataset,num_layers=[4,5,6],optimizer = ['RMSprop', 'Adam', 'Adamax', 'Nadam'],regularizer=[None,l1(0.001),l2(0.001)]):
    
    ###Creation des différents dossier contenant les plots###
   
    parametrage= "parametrage"
    # Parent Directory path
    parent_dir = os.getcwd()
    # Path
    path1 = os.path.join(parent_dir+'/Plots', parametrage)
    path2 = os.path.join(parent_dir+'/Excel', parametrage)
    if os.path.exists(path1):
    # removing the file using the os.remove() method
        shutil.rmtree(path1)
    if os.path.exists(path2):
    # removing the file using the os.remove() method
        shutil.rmtree(path2)
    os.mkdir(path1)
    os.mkdir(path2)
    print("Directory '% s' created" % parametrage)
    print("Plusieurs paramètrage sont en cours, elle permettront de trouver un réseau optimal \n")
    print("Les courbes et tableurs excel seront enregistrées dans les sous-dossier suivant : parametrage \n")
    print('=' * term_size.columns)
    
    
    ###Début d'optimisation selon la première phase###
    
    print("###Optimisation du nombre de neurones , de couches et de l'optimiseur###\n")
    neurons=500 #On commance par un nombre de neurone relativement bas. 
    print("Le nombre de neurones de la première couche est 500")
    layers=create_layer(max(num_layers),neurons,config.n_classes)
    layers=layers[min(num_layers)-1:] #On ne récupère que les nombres de couches pertinant
    result=tuning_layers(layers,optimizer) #fonction d'optimisation de la première étape
    while result['mean_test_score'].iloc[0]<0.10: #Si le résultat n'est pas interessant , on essaye avec plus de neurones dans chaque couche 
        print("Echec d'un résultat viable avec une couche d'entrée de %s, on test avec %s"%(neurons,2*neurons))
        neurons=neurons*2
        layers=create_layer(7,neurons)
        result=tuning_layers(layers)
        plt.show()
    print('=' * term_size.columns)
    result.to_pickle("result_test.pkl") #Enregistrement du premier résultat , on ne garde que les 5 meilleurs
    print('=' * term_size.columns)
    
    
    ###Début d'optimisation de la seconde phase###
    
    print("###Optimisation du regularisateur ###\n")
    coefficient=0.001
    result_2=tuning_regularizer(result,regularizer,coefficient)
    
    while result['mean_test_score'].iloc[0]<0.70: #Si le résultat n'est pas interessant, on essaie avec un coefficient de régularisation plus élevé. 
        if coefficient ==0.005:
            print("Pas de réseau viable trouvé , tant pis , le processus continue...\n")
            break
        print("Echec d'un résultat viable avec un coefficient de régularisation  de %s, on test avec %s"%(coefficient,coefficient+0.001))
        for i in regularizer: 
            if ('l1' in dir(i)) and ('l2' in dir(i)):
                i.l1+=0.001
                i.l2+=0.001
                print("Les coefficients sont %s et %s "%(i.l1,i.l2))
            elif 'l1' in dir(i):
                i.l1+=0.001
                print("Le coefficient devient %s" %i.l1)
              
            if 'l2' in dir(i):
                i.l2+=0.001
                print("Le coefficient devient %s" %i.l2)
        coefficient+=0.001
        result_2=tuning_regularizer(result,regularizer)

        
    plt.show()
    return result_2