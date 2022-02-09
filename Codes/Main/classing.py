# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:00:20 2022

@author: Antony
Crée les classes qui entraîneront le modèle et effectue une homogénisation de celle-ci afin
d'éviter des classes avec des effectifs trop conséquent ce qui biaiserait les poids
Sortie: Tableau prêt à être implanter dans un réseau
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import statistics
from stats import overview
def classing_input(dataset): 
    """
    Classe selon un chiffre entier chaque vecteur selon la proximité de son temps associé avec 
    un multiple de 10. 26->classe 3 (30ms)
    Elle définie les classes et devrait donc s'adapter au set, ici 10 ,c'est le plus cohérent.
    """
    print("Creation des classes en cours ...\n")
    dataset['time']=[round(dataset['time'].iloc[i]/10) for i in tqdm(range(len(dataset['time'])))] #on redistribue le temps sous forme d'interval.
    return dataset

def homogenisation(dataset):
    """
    Garde les classes les plus interessantes , celles qui ont un effectif conséquent mais pas trop
    non plus. Cette fonction est malheureusement arbitraire pour l'instant....
    """
    
    fig, (ax_class, ax_eff, ax_eff2) = plt.subplots(3, 1)
    ### On récupère les effectifs de chaque classe###
    keep_classes = {}
    for i in dataset['time'].drop_duplicates().sort_values():
        keep_classes[i] = len(dataset[dataset['time'] == i])
    ax_class.plot(keep_classes.keys(), keep_classes.values(), 'ro')
    find_quantile = list(keep_classes.values())
    q75, q25 = np.percentile(find_quantile, [75, 25])
    ### On ne garde que les classes avec un effectif supérieur à 75 % du set ###
    keep_classes = {k: keep_classes[k]
        for k in keep_classes.keys() if keep_classes[k] > q75}
    
    dataset = dataset[dataset['time'].isin(list(keep_classes.keys()))] #on conserve les élèments avec les classes sélectionnées
    ###Affichage###
    ax_class.plot(keep_classes.keys(), keep_classes.values(), 'bo')
    ax_class.set_xlabel("classes")
    ax_class.set_ylabel("effectif")
    ax_class.set_title("On ne garde que les classes en bleu")
    ### On réduit maintenant les électifs des classes à la population trop nombreuse###
    find_quantile = list(keep_classes.values())
    ax_eff.boxplot(find_quantile)
    ax_eff.set_title("Quartiles initial des effectifs composant nos classes ")
    ax_eff.set_ylabel("effectif")
    q40, q60 = np.percentile(find_quantile, [40, 60])
    intr_qr = q60-q40
    maximum = int(q60+(1.5*intr_qr)) #Borne supérieur à ne pas dépasser
    print("Identification des classes conséquentes (> 60% du set)...\n")
    ###On identifie les plus grosses classes ###
    too_big = {k: keep_classes[k]
        for k in tqdm(keep_classes.keys()) if keep_classes[k] > q60}
    ###On réduit leur effectif à la borne maximum###
    print("Réduction des effectifs des classes trop conséquentes ...\n")
    for i in tqdm(too_big.keys()):
        if too_big[i] > maximum:
            dataset[dataset['time'] == i] = dataset[dataset['time']
                == i].sample(maximum)  # on ne garde qu'un echantillion
    dataset = dataset.dropna()  # important
    print("Actualisation des nouveaux effectifs pour chaque classe ...\n")
    for i in tqdm(dataset['time'].drop_duplicates().sort_values()):
        keep_classes[i] = len(dataset[dataset['time'] == i])
    find_quantile = list(keep_classes.values())
    L=list(keep_classes.values())
    maximum,minimum,mean=max(L),min(L),statistics.mean(L)
    print("Les effectifs minimum ,maximum et la moyenne sont %d , %d et %d respectivement \n"%(minimum,maximum,mean) )
    ax_eff2.boxplot(find_quantile)
    ax_eff2.set_title("Quartile des effectifs revisités")
    ax_eff2.set_ylabel("effectif")
    plt.tight_layout()  # pour rendre le plot plus beau et éviter les dépassement en écriture
    fig.savefig("Plots/Profil de nos classes")
    return dataset
def classifier(dataset):
    """
    Main function
    """
    dataset=classing_input(dataset)
    dataset=homogenisation(dataset)
    print("Le set devient : \n",dataset)
    overview(dataset.drop('index',axis=1),"Profil du set finalisé")
    return dataset