# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 20:22:44 2022

@author: Antony
"""

if __name__ == '__main__' :
    import os
    import sys
    from pathlib import Path
    from load import load
    from vectorization import vectorization
    from reduction import reduction 
    import shutil
    if len(sys.argv) > 3 or len(sys.argv)<=1 :
        print("Mauvaise entrée, de la forme : python neural_genarator.py xlsx_file pkl_file")
        sys.exit()
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
    print("Début de la première étape : Chargement des données \n")
    xlsx_file = Path(sys.argv[1])
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
    print("Début de la seconde étape : Vectorisation (on ne garde que les élèments à une boutique)\n")
    data=vectorization(data) 
    data.to_pickle("DataFrame/Pré-processing/data_vectorized.pkl")
    data.to_excel("Excel/Pré-processing/data_vectorized.xlsx")

    print()
    print("Ce tableau est enregistré sous le nom de data_vectorized\n")
    ###Réduction###
    print("Début de la troisième étape : Réduction en vecteurs uniques \n")
    data,liste_temps=reduction(data) 
    liste_temps.to_pickle("DataFrame/Pré-processing/liste_temps.pkl")
    data.to_pickle("DataFrame/Pré-processing/data_reduced.pkl")
    data.to_excel("Excel/Pré-processing/data_reduced.xlsx")
    print()
    print("Ce tableau est enregistré sous le nom de  data_reduced \n")
    
