# -*- coding: utf-8 -*-

"""
2nd étape:
On ne peut pas implanter des intervals dans un réseau de neurones,
cet algorithme aspire à transformer un interval [4,8) par exemple , 
en un simple chiffre : 3 , la somme des entités. Il prend donc en entrée le tableau généré 
par load.py, c'est la seconde étape. Chaque élèment du tableau est une liste dans le cas ou 
on aurait par la suite affaire à des clusters.
Sortie : Un tableau
@author: Antony
"""
from datetime import date
import pandas as pd
from tqdm import tqdm
def isnan(num):
    return num != num
def dated(char):#retourne le nombre de jour évalués , on perd en information 
    """
    Quand on rencontre une date , on la traduit en nombre de jour
    """
    begin = char.find("[")
    middle = char.find(",")
    end = char.find(")")
    dates = []
    cursor = char.find('/')
    number = int(char[begin+1:cursor])
    dates.append(number)
    next_cursor = 0
    while next_cursor !=-1:
        next_cursor=char[cursor+1:].find('/')
        if next_cursor == -1:
            number = int(char[cursor+1:end])
            dates.append(number)
        else:
            number = char[cursor+1:cursor + next_cursor+1]
            if ',' in number :
                middle= number.find(',')
                dates.append(int(number[:middle]))
                dates.append(int(number[middle+1:]))
            else: 
                number = int(number)
                dates.append(number)
        cursor += next_cursor+1
    f_date = date(dates[2], dates[1], dates[0])
    l_date = date(dates[5], dates[4], dates[3])
    delta = l_date - f_date
    return delta.days

def intervals_to_coordinates(char):
    """
    Pour traiter les intervals conventionnels
    """
    begin = char.find("[")
    middle = char.find(",")
    end = char.find(")")
    if "/" in char : 
        return dated(char)
    else:
        number_1 = char[begin+1:middle]
        number_2 = char[middle+1:end]
        number_1,number_2=int(number_1),int(number_2)
        coor = number_2-number_1
        return coor
def intervals_to_coordinates_id(char): 
    """
    On récupère l'identifiant de chaque élèment
    """
    begin = char.find("[")
    middle = char.find(",")
    number_1 = char[begin+1:middle]
    return int(number_1)
def intervals_to_coordinates_var(char):
    """
    On récupère la variable de chaque élèment
    """
    begin = char.find("[")
    middle = char.find("]")
    number_1 = char[begin+1:middle]
    return int(number_1)
def list_to_coordinates(L): 
    """
    On transforme la liste de coordoonnées pour une dimention en liste de chiffre
    """
    translated=[]
    if L is None or isnan(L)==True : 
        return 0
    for i in L:
        translated.append(intervals_to_coordinates(i))
    return sum(translated) #ici on renvoie 0 car on ne s'occupe pas de cluster
def list_to_coordinates_id(L): 
    translated=[]
    if L is None or isnan(L)==True : 
        return 0        
    for i in L:
        translated.append(intervals_to_coordinates_id(i))
    #return translated
    return translated[0]

def list_to_coordinates_var(L):
    translated=[]
    if L is None or isnan(L)==True : 
        return 0
    for i in L:
        translated.append(int(i))
    return translated[0]
#i want to do smth like :
    #df_1=df_1[f(df_1)==r] but it doesn't work due to vectors...
def vectorization(df_1):
    """
    Main function
    Sur une boucle on transforme à un à un chaque colonne du tableau
    """
    df_2=pd.DataFrame(list(map(list_to_coordinates,list(df_1[1]))))
    df_2=df_2[df_2[0]==1] # keeping only data with one shop , it keeps the right index
    df_1=df_1.iloc[df_2.index]
    data_1= pd.DataFrame(list(df_1['time']),index=list(df_1.index))
    
    print("Vectorisation en cours \n")
    for i in tqdm(df_1.columns[:-1]):
        #print()
        if i ==1:
            data_1[i]= list(map(list_to_coordinates_id,list(df_1[i])))
        elif i == 'var': #Attention , c'est elif !
            data_1[i] = list(map(list_to_coordinates_var,list(df_1[i])))
        else: 
            h= list(map(list_to_coordinates,list(df_1[i]))) #don't forget the iloc !
            data_1[i]=h
    data_1=data_1.rename(columns = {0 : 'time'})
    print()
    print("Le tableau devient \n",data_1)

    return data_1

