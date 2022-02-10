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
    from tuning import tuning
    if len(sys.argv) != 2  :
        print("Mauvaise entrée, de la forme : python neural_genarator.py xlsx_file pkl_file")
        sys.exit()
    ###Nettoyage###
    data = pd.read_pickle(sys.argv[1])
    data=data[data.columns[:9]] #Ne pas oublier ! 
    print("Début de la septième étape : Paramétrage \n")
    tuning(data)
