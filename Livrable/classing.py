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
import warnings 
warnings.filterwarnings('ignore')

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
    
    fig1, (ax_class, ax_eff) = plt.subplots(2, 1)
    fig2,(ax_eff2,ax_class_2) = plt.subplots(2, 1)
    
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
    
    find_quantile = list(keep_classes.values())

    ###Affichage###
    ax_class.plot(keep_classes.keys(), keep_classes.values(), 'bo')
    ax_class.set_xlabel("classes")
    ax_class.set_ylabel("effectif")
    ax_class.set_title("On se base sur ces classes en bleu")
    ax_class.set_ylim([0,max(find_quantile)+20])
    
    ###Quartiles initial###
    ax_eff.boxplot(find_quantile)
    ax_eff.set_title("Etendue initial des effectifs composant nos classes ")
    ax_eff.set_ylabel("effectif")
    ax_eff.set_ylim([0,max(find_quantile)+20])
    fig1.tight_layout()
    
    ###On identifie les plus grosses classes ###

    q40, q60 = np.percentile(find_quantile, [40, 60])
    intr_qr = q60-q40
    maximum = int(q60+(1.5*intr_qr)) #Borne supérieur à ne pas dépasser
    print("Identification des classes conséquentes (> 60% du set)...\n")
    
    too_big = {k: keep_classes[k]
        for k in tqdm(keep_classes.keys()) if keep_classes[k] > q60}
    
        #On réduit leur effectif à la borne maximum#
    print("Réduction des effectifs des classes trop conséquentes ...\n")
    for i in tqdm(too_big.keys()):
        if too_big[i] > maximum:
            dataset[dataset['time'] == i] = dataset[dataset['time']
                == i].sample(maximum)  # on ne garde qu'un echantillion
            
    ###On ajoute des 'dupliqués' au classes trop peu nombreuses###
    dataset = dataset.dropna()  # important
    for i in dataset['time'].drop_duplicates().sort_values():
        if i not in keep_classes.keys():
            to_add=[dataset[dataset['time']==i]]
            to_add_int=to_add.copy()
            if 2*len(to_add_int[0])<q75/2:
                while len(to_add[0])<q75/2:
                    dataset=dataset.append(to_add_int)
                    to_add=[dataset[dataset['time']==i]]
    print("Actualisation des nouveaux effectifs pour chaque classe conséquente ...\n")
    for i in tqdm(dataset['time'].drop_duplicates().sort_values()):
        keep_classes[i] = len(dataset[dataset['time'] == i])
        
    ###Quartiles###
    find_quantile = list(keep_classes.values())
    maximum,minimum,mean=max(find_quantile),min(find_quantile),statistics.mean(find_quantile)
    print("Les effectifs minimum ,maximum et la moyenne sont %d , %d et %d respectivement \n"%(minimum,maximum,mean) )
    ax_eff2.boxplot(find_quantile)
    ax_eff2.set_title("Etendu des effectifs du nouveau profil")
    ax_eff2.set_ylabel("effectif")
    ax_eff2.set_ylim([0,maximum+20])

    ###Etendue du nouveau profil###
    ###Profil des classes###
    ax_class_2.plot(keep_classes.keys(), keep_classes.values(),'go')
    ax_class_2.set_ylim([0,maximum+20])
    ax_class_2.set_xlabel("classes")
    ax_class_2.set_ylabel("effectif")
    ax_class_2.set_title("Nouveau profil")
    fig2.tight_layout()  # pour rendre le plot plus beau et éviter les dépassement en écriture
    ###Sauvegarde###
    fig1.savefig("Plots/Pré-processing/Lissage_classes/Profil_initial_de_nos_classes")
    fig2.savefig("Plots/Pré-processing/Lissage_classes/Nouveau_profil_de_nos_classes")

    return dataset
def classifier(dataset):
    """
    Main function
    """
    dataset=classing_input(dataset)
    #dataset=homogenisation(dataset)
    print("Le set devient : \n",dataset)
    overview(dataset.drop('index',axis=1),"Profil du set finalisé")
    return dataset

