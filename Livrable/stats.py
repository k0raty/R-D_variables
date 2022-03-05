# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 22:25:22 2022

@author: Antony
Fichier réunnissant différentes fonctions statistiques
"""



import pandas as pd
import matplotlib.pyplot as plt 
#Creating subplot of each column with its own scale
def overview(sheet,title):
    """
    Permet d'avoir une vision d'ensemble du profil des données
    """
    print("Dessin du profil statistique  ... \n")

    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
    fig, axs = plt.subplots(1, len(sheet.columns), figsize=(20,10))
    
    for l, ax in enumerate(axs.flat):
        ax.boxplot(pd.to_numeric(sheet.dropna().iloc[:,l]), flierprops=red_circle) #drop nan values
        ax.set_title(sheet.columns[l], fontsize=20, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig("Plots/Pré-processing/"+title)
