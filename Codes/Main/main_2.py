# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:26:22 2022

@author: Antony
"""

if __name__ == '__main__' :
   
    import sys
    import pandas as pd
    from prunning import smooth
    import matplotlib.pyplot as plt 
    from classing import classifier
    from overview import overview
    if len(sys.argv) != 3  :
        print("Mauvaise entrée, de la forme : python neural_genarator.py xlsx_file pkl_file")
        sys.exit()
    ###Nettoyage###
    print("Début de la quatrième étape : Nettoyage \n")
    data = pd.read_pickle(sys.argv[1])
    liste_temps=pd.read_pickle(sys.argv[2])
    data,security=smooth(data,liste_temps)
    data.to_pickle("DataFrame/data_prunned.pkl")
    print()
    print("Ce tableau est enregistré dans data_prunned.pkl \n")
    plt.draw()
    plt.pause(1)
    ###Classification###
    print("Début de la cinquième étape : Classification \n")    
    data=classifier(data)
    data.to_pickle("DataFrame/data_classed.pkl")
    print()
    print("Ce tableau est enregistré dans data_classed.pkl \n")
    print("Le set est maintenant prêt \n")
    plt.show()
 