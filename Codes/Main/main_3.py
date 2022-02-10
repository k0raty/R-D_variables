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
    from tuning import tuning 
    if len(sys.argv) != 2  :
        print("Mauvaise entrée, de la forme : python neural_genarator.py xlsx_file pkl_file")
        sys.exit()
    ###Nettoyage###
    data = pd.read_pickle(sys.argv[1])
    print("Début de la sixième étape : Pré-paramétrage \n")
    data=data[data.columns[:9]] #Ne pas oublier ! 
    num_layers,optimizer,regularizer=overview(data)
    print("Début de la septième étape : Paramétrage \n")
    tuning(data,num_layers,optimizer,regularizer)
    
