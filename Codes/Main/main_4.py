# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:23:24 2022

@author: Antony
"""
# create layers

if __name__ == '__main__' :
    
    import sys
    import pandas as pd
    import matplotlib.pyplot as plt 
    import numpy as np
    import config
    from tuning_layers import tuning_layers
    from tensorflow.keras.utils import to_categorical
    from config import create_layer 
    if len(sys.argv) != 2  :
        print("Mauvaise entrée, de la forme : python neural_genarator.py xlsx_file pkl_file")
        sys.exit()
    ###Nettoyage###
    data = pd.read_pickle(sys.argv[1])
    data=data[data.columns[:9]] #Ne pas oublier ! 
    print("Début de la septième étape : Paramétrage \n")
    neurons=500
    layers=create_layer(7,neurons,config.n_classes)

    result=tuning_layers(layers)
    while result['mean_test_score'].iloc[0]<0.60:
        print("Echec d'un résultat viable avec une couche d'entrée de %s, on test avec %s"%(neurons,2*neurons))
        neurons=neurons*2
        layers=create_layer(7,neurons)
        result=tuning_layers(layers)
        plt.show()
