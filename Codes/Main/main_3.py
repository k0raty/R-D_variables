# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 14:08:39 2022

@author: Antony
"""

if __name__ == '__main__' :
   
    import sys
    import pandas as pd
    import matplotlib.pyplot as plt 
    import config
    from overview_2 import overview
    from tuning_layers import tuning_layers
    from config import create_layer
    if len(sys.argv) != 2  :
        print("Mauvaise entrée, de la forme : python neural_genarator.py xlsx_file pkl_file")
        sys.exit()
    ###Nettoyage###
    data = pd.read_pickle(sys.argv[1])
    print("Début de la sixième étape : Pré-paramétrage \n")
    data=data[data.columns[:9]] #Ne pas oublier ! 
    num_layers,optimizer,regularizer=overview(data)
    print("Début de la septième étape : Paramétrage \n")
    neurons=500
    print("Le nombre de neurones de la première couche est 500")
    layers=create_layer(max(num_layers),neurons,config.n_classes)
    layers=layers[min(num_layers)-1:] #On ne récupère que les nombres de couches pertinant
    result=tuning_layers(layers)
    while result['mean_test_score'].iloc[0]<0.60:
        print("Echec d'un résultat viable avec une couche d'entrée de %s, on test avec %s"%(neurons,2*neurons))
        neurons=neurons*2
        layers=create_layer(7,neurons)
        result=tuning_layers(layers)
        plt.show()

    
