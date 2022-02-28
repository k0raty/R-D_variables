# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:57:45 2022

@author: Antony
"""

from overview_LR import overview_LR
from overview_regularizer import overview_regularizer
from overview_layer import overview_layer
import os
import shutil
term_size = os.get_terminal_size()
def overview(dataset):
    Pre_parametrage= "Pré-paramétrage"
    
    # Parent Directory path
    parent_dir = os.getcwd()
    # Path
    path1 = os.path.join(parent_dir+'/Plots', Pre_parametrage)
    path2 = os.path.join(parent_dir+'/Excel', Pre_parametrage)
    path3 = os.path.join(parent_dir+'/DataFrame', Pre_parametrage)
    if os.path.exists(path1):
    # removing the file using the os.remove() method
        shutil.rmtree(path1)
    if os.path.exists(path2):
    # removing the file using the os.remove() method
        shutil.rmtree(path2)
    if os.path.exists(path3):
    # removing the file using the os.remove() method
        shutil.rmtree(path3)
    os.mkdir(path1)
    os.mkdir(path2)
    os.mkdir(path3)

    print("Directory '% s' created" % Pre_parametrage)
    print("La fonctions d'activation des couches cachées est 'relu' car le réseau est un multicouche non récursif, la fonction de sortie est 'Softmax' pour un classifacteur multiple \n")
    print("Plusieurs approches de paramètrage sont en cours, elle permettront d'orienter le programme dans la prochaine étape \n")
    print("Les courbes et tableurs excel seront enregistrées dans les sous-dossier suivant : Pre_parametrage \n")
    
    print('=' * term_size.columns)
    print("###Approche du nombre de couches###\n")
    num_layers=overview_layer(dataset)
    n_layers=min(num_layers)
    
    print('=' * term_size.columns)
    print("###Approche de l'optimiseur###\n")
    optimizer=overview_LR(dataset,n_layers)
    
    print('=' * term_size.columns)
    print("###Approche du régularisateur###\n")
    print("Pour cette approche on utilise l'optimiseur  : %s pour le réseaux aux nombre de couche suivant: %s "%(optimizer[0],n_layers))
    regularizer=overview_regularizer(dataset,optimizer[0],n_layers)
    
    print("Finalement notre recherche s'orientera sur les paramètres suivant: %s, %s, et %s \n" %(num_layers,optimizer,regularizer))
    return num_layers,optimizer,regularizer    