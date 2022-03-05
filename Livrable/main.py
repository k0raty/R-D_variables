#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:26:53 2022

@author: k0raty
"""

if __name__ == '__main__' :
    import os
    import sys
    from load import load
    from vectorization import vectorization
    from reduction import reduction 
    import shutil
    from prunning import smooth
    from classing import classifier
   
    
  

    if len(sys.argv) > 3 or len(sys.argv)<=1 :
        print("Mauvaise entrée, de la forme : python main.py xlsx_file.xlsx pkl_file.pkl")
        sys.exit()
    term_size = os.get_terminal_size()
    ###Création des dossiers###
    DataFrame,Plots,Excel,Processing = "DataFrame","Plots","Excel","Pré-processing"
    
    # Parent Directory path
    parent_dir = os.getcwd()
    # Path
    path1,path2,path3 = os.path.join(parent_dir, DataFrame),os.path.join(parent_dir,Plots),os.path.join(parent_dir,Excel)
    if os.path.exists(path1):
    # removing the file using the os.remove() method
        shutil.rmtree(path1)
    if os.path.exists(path2):
    # removing the file using the os.remove() method
        shutil.rmtree(path2)    # Create the directory
    if os.path.exists(path3):
    # removing the file using the os.remove() method
        shutil.rmtree(path3)    # Create the directory
    os.mkdir(path1)
    os.mkdir(path2)
    os.mkdir(path3)
    ###Création du sous-dossier Pré-parmétrage### 
    path4 = os.path.join(parent_dir+'/Plots', Processing)
    path5 = os.path.join(parent_dir+'/Excel', Processing)
    path6 = os.path.join(parent_dir+'/DataFrame', Processing)
    if os.path.exists(path4):
    # removing the file using the os.remove() method
        shutil.rmtree(path4)
    if os.path.exists(path5):
    # removing the file using the os.remove() method
        shutil.rmtree(path5)    # Create the directory
    if os.path.exists(path6):
    # removing the file using the os.remove() method
        shutil.rmtree(path6)    # Create the directory
    os.mkdir(path4)
    os.mkdir(path5)
    os.mkdir(path6)
    print("Directory '% s' created" % Processing)
    

    ###Chargement###
    print("Tableur excel , fichier pkl et png seront enregistrés dans les dossier Excel , DataFrame et Plots respectivement \n")
    print("Il y aura plusieurs étapes, tout d'abord des étapes de pré-processing. Les données seront donc enregistrées dans ce sous-dossier\n")
    #Séparateur entre étape#
    print('#' * term_size.columns)

    print("Début de la première étape : Chargement des données \n")
    xlsx_file = parent_dir + "/"+ sys.argv[1]
    if len(sys.argv) == 3:
        pkl_file = sys.argv[2]
        data=load(xlsx_file,pkl_file)
    else:
        data=load(xlsx_file)
    data.to_pickle("DataFrame/Pré-processing/data.pkl")
    data.to_excel("Excel/Pré-processing/data.xlsx")
    print()
    print("Ce tableau est enregistré sous le nom de data \n")
    ###Vectorisation###
    print('#' * term_size.columns)
    print("Début de la seconde étape : Vectorisation (on ne garde que les élèments à une boutique)\n")
    data=vectorization(data) 
    n_vector=len(data.columns)
    data.to_pickle("DataFrame/Pré-processing/data_vectorized.pkl")
    data.to_excel("Excel/Pré-processing/data_vectorized.xlsx")

    print()
    print("Ce tableau est enregistré sous le nom de data_vectorized\n")

    ###Réduction###
    print('#' * term_size.columns)
    print("Début de la troisième étape : Réduction en vecteurs uniques \n")
    data,liste_temps=reduction(data)
    liste_temps.to_pickle("DataFrame/Pré-processing/liste_temps.pkl")
    data.to_pickle("DataFrame/Pré-processing/data_reduced.pkl")
    data.to_excel("Excel/Pré-processing/data_reduced.xlsx")
    print()
    print("Ce tableau est enregistré sous le nom de  data_reduced \n")

    ###Création du sous-sous-dossier Lissage_coordonnées
    Lissage='Lissage_coordonnées'
    parent_dir = os.getcwd()

    path1 = os.path.join(parent_dir+"/Plots/Pré-processing",Lissage)
    if os.path.exists(path1):
    # removing the file using the os.remove() method
        shutil.rmtree(path1)

    os.mkdir(path1)

    print("Directory '% s' created" % Lissage)
    
    ###Création du sous-sous-dossier Lissage_classes###
    
    Lissage='Lissage_classes'
    parent_dir = os.getcwd()

    path1 = os.path.join(parent_dir+"/Plots/Pré-processing",Lissage)
    if os.path.exists(path1):
    # removing the file using the os.remove() method
        shutil.rmtree(path1)

    os.mkdir(path1)

    print("Directory '% s' created" % Lissage)
    
    ###Nettoyage###
    print('#' * term_size.columns)
    print("Début de la quatrième étape : Nettoyage \n")
    data,security=smooth(data,liste_temps)
    data.to_pickle("DataFrame/Pré-processing/data_prunned.pkl")
    data.to_excel("Excel/Pré-processing/data_prunned.xlsx")
    
    print()
    print("Ce tableau est enregistré sous le nom de data_prunned \n")
   
    ###Classification###
    
    print('#' * term_size.columns)

    print("Début de la cinquième étape : Classification \n")  
    data=classifier(data)
    data.to_pickle("DataFrame/Pré-processing/data_classed.pkl")
    data.to_excel("Excel/Pré-processing/data_classed.xlsx")

    print()
    print("Ce tableau est enregistré sous le nom de data_classed \n")
    print("Le set est maintenant prêt \n")
   
    #####NOUVEAU IMPORT APRES ENREGISTREMENTS#####
    from overview import overview
    from tuning import tuning 
    from model_display import model_display
    from statistical_comparison import stat_comp
    from tuning_time import tuning_time
    from evaluation import evaluation 
    import tensorflow as tf
    ##############################################
    
    ###Pré-paramétrage###
    
    print('#' * term_size.columns)
    print("Début de la sixième étape : Pré-paramétrage \n")
    print("Des graphiques sur les performances des réseaux sont enregistrés à chaque sous-étape \n")
    
    data = data[data.columns[:n_vector]] #C'est le set final sans les informations statistiques
    num_layers,optimizer,regularizer=overview(data)
    
    ###Paramétrage###
    
    print('#' * term_size.columns)
    print("Début de la septième étape : Paramétrage \n")
    print("A chaque sous-étape sera enregistré un graphique ou une heatmap associée dans /Plots/Paramétrage \n")
    results_df,n_train,n_test=tuning(num_layers,optimizer,regularizer)
    print("Résultats enregistrés sous le nom de Opt_batch(.pkl/.xlsx)")
    results_df.to_pickle("DataFrame/Paramétrage/Opt_batch.pkl")
    results_df.to_excel("Excel/Paramétrage/Opt_batch.xlsx")
    Post_parametrage="Post_paramétrage"
      
    parent_dir = os.getcwd()
    # Path
    path1 = os.path.join(parent_dir+'/Plots', Post_parametrage)
    path2 = os.path.join(parent_dir+'/Excel', Post_parametrage)
    path3 = os.path.join(parent_dir+'/DataFrame', Post_parametrage)
    
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
    
    
    print("Résultats enregistrés sous le nom de Opt_batch(.pkl/.xlsx)")
    results_df.to_pickle("DataFrame/Paramétrage/Opt_batch.pkl")
    results_df.to_excel("Excel/Paramétrage/Opt_batch.xlsx")
    
    print("Ceci est la phase de post-paramétrage et son sous dossier respectif à été crée \n")
    ###Scabilité et performances###
    print('#' * term_size.columns)
    print("Début de la huitième étape : Dessins \n")
    print("Dessin du profil statistique des deux meilleurs modèles \n")
    model_display(results_df)
    
    ###Etude statistique###
    print('#' * term_size.columns)
    print("Début de la neuvième étape : Hypothèses statistiques \n")
    print("Comparaison statistique des meilleurs modèles\n")
    results_df= stat_comp(results_df,n_train,n_test)
    results_df.to_pickle("DataFrame/Post_paramétrage/best_models.pkl")
    results_df.to_excel("Excel/Post_paramétrage/best_models.xlsx")
    print("Les modèles sélectionnés sont enregistrés dans best_models.pkl et best_models.xlsx \n")
    ###Comparaison des temps et évaluation du modèle ###
    
        #Dossier Evaluation#
    Post_parametrage="Evaluation"
      
    # Path
    path1 = os.path.join(parent_dir, Post_parametrage)
    
    if os.path.exists(path1):
    # removing the file using the os.remove() method
        shutil.rmtree(path1)
    
    os.mkdir(path1)
    
    print('#' * term_size.columns)
    print("Début de la dixième étape : Comparaisons temporelles \n")
    print("Comparaison des temps des modèles sélectionnés\n")
    tuning_time(results_df,security)
    
   
 
