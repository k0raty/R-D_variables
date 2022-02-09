# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:57:45 2022

@author: Antony
"""

from overview_LR import overview_LR
from overview_regularizer import overview_regularizer
from overview_batch import overview_batch
import matplotlib.pyplot as plt 
import time 
from tqdm import tqdm
import numpy as np
import keyboard 
import os
term_size = os.get_terminal_size()
def overview(dataset):
    print("Plusieurs premières approches de paramètrage sont en cours, elle permettront d'orienter l'utilisateur dans la prochaine étape \n")
    #overview_LR(dataset)
    #plt.show()
    num_layers = [] #Liste du nombre de couche à évaluer pour l'étape d'après 
    current_time=time.time()  
    anyone=0 #to check if someone is here 
    print('=' * term_size.columns)
    print("###Approche du régularisateur###\n")
    print("Appuyez sur p pour commencer la saisie , si pas de réponse d'ici 30 secondes, la prochaine étape sera lancée avec des paramètres par défault \n")
    ###progress bar###
    LENGTH = 30 # Number of iterations required to fill pbar
    pbar = tqdm(total=LENGTH) # Init pbar
    pbar.set_description("Time left")
    ###waiting loop###
    while time.time() - current_time < 30:
        time.sleep(1)
        pbar.update(n=1)
        if keyboard.is_pressed('p'):
            anyone=1
            print('p is pressed \n')
            break
    pbar.close()
    if anyone == 1 : #si qqun à rentré des informations
        print("Veuillez saisir le nombre de couches à évaluer sous la forme : 4 8 3 1 6 ... , 8 c'est beaucoup \n")
        num_layers = np.array([int(item) for item in input("Couche à évaluer ?: ").split()])
        print("Les couches sont : ",num_layers)
        optimizer = input("Quel optimisateur utiliser ? (Les plus connus étant : sgd, rmsprop, adagrad, adam ) : ")
        print("L'optimiseur est : ",optimizer)
        overview_regularizer(dataset,optimizer,num_layers)
        plt.show()
    
    if anyone == 0:
        print("Aucune entrée : paramètres par défaut saisis \n")
        print("Les couches sont : [0,1,2,3,4] et l'optimiseur est adam")
        overview_regularizer(dataset)
        plt.show()
    ###Batch###  
    anyone=0
    print('=' * term_size.columns)
    print("###Approche du batch###\n")
    print("Appuyez sur p pour commencer la saisie , si pas de réponse d'ici 30 secondes, la prochaine étape sera lancée avec des paramètres par défault \n")
    ###progress bar###
    LENGTH = 30 # Number of iterations required to fill pbar
    pbar = tqdm(total=LENGTH) # Init pbar
    pbar.set_description("Time left")
    ###waiting loop###
    while time.time() - current_time < 30:
        time.sleep(1)
        pbar.update(n=1)
        if keyboard.is_pressed('p'):
            anyone=1
            print('p is pressed \n')
            break
    pbar.close()
    if anyone == 1 : #si qqun à rentré des informations
        n_layers=input("Veuillez saisir le nombre de couches du réseau à entraîner :\n")
        print("Le nombre de couches est : ",n_layers)
        optimizer = input("Quel optimisateur utiliser ? (Les plus connus étant : sgd, rmsprop, adagrad, adam ) : ")
        print("L'optimiseur est : ",optimizer)
        overview_batch(dataset,optimizer,n_layers)
        plt.show()
    
    if anyone == 0:
        print("Aucune entrée : paramètres par défaut saisis \n")
        print("Le nombre de couche est : 3 , l'optimiseur est adam et le regularisateur :l2(0.001)")
        overview_batch(dataset)
        plt.show()