    # -*- coding: utf-8 -*-

"""
3ème étape: 
Pour réduire le nombre de vecteurs identiques en un unique vecteurs sinon cela engendrerait
des complications lors de l'entraînement du réseau.
Sortie:
    Un tableau ainsi qu'une liste de temps qui contient les temps utilisés pour chaque
    vecteurs afin d'établir les moyennes.
Compléxité élevée :( , peut sûrement se diminuer en entrant les commandes via excel et non python. 
@author: Antony

"""
import statistics
import pandas as pd
import math 
from tqdm import tqdm
import numpy as np
def reduction(sheet):
        """
        Main function
        Procéde à la réduction du tableau en vecteurs uniques
        """
        sheet_2=sheet.drop(['time'],axis=1)
        visited=[sheet_2.iloc[0].values] #Liste qui contiendra les vecteurs déjà rencontré
        temps=[] #Liste des temps qui seront moyennés pour chaque vecteur
        index=[[0]]
        temps.append([float(sheet['time'].iloc[0])])
        print("Reduction en cours :\n")
        for i in tqdm(range(1,len(sheet))):
             
             h=0
             while  (h<len(visited) and (sheet_2.iloc[i].values == visited[h]).all() == False) : #ici ,c'est une méthode pour pour voir si un vecteur est dans cette liste de vecteurs. 
                 h+=1
             if h == len(visited):
                 visited.append(sheet_2.iloc[i].values) #le .values fait que c'est un vecteur. 
                 temps.append([float(sheet['time'].iloc[i])])
                 index.append([int(sheet_2.iloc[i].name)])
             else:
                 temps[h].append(float(sheet['time'].iloc[i]))
                 index[h].append(int(sheet_2.iloc[i].name)) #on conserve l'indice du point regroupé
        info=[] #corresponds aux quartiles temps de chaque vecteur ainsi qu'à l'écart-type,utile pour l'étape suivante
        print()
        print("Statistiques des temps : \n")
        for i in tqdm(temps) : #add data that need a len >2
             if len(i) >2:
                 quantiles  = np.quantile(i,[0.25,0.5,0.75]) #statistics.quantiles fonctionne sur la version 3.8 de python
                 info.append([math.sqrt(statistics.variance(i)),quantiles[0],quantiles[2]])
             else : 
                 info.append(3*[0])
            
            
        info_temps=[(statistics.mean(i),min(i),max(i),statistics.median(i),len(i)) for i in temps] # regroupe d'autres informations sur le temps
        columns=['mean','min','max','median','lenght','écart-type','25%','75%']
        df_info = pd.DataFrame(info)
        df_temps=pd.DataFrame(info_temps)
        df_temps=pd.concat([df_temps,df_info],axis=1)
        df_temps.columns=columns   
        df=pd.DataFrame(visited)
        df=pd.concat([df_temps['mean'],df],axis=1)
        df.columns=sheet.columns
        df=pd.concat([df,df_temps.drop(['mean'],axis=1)],axis=1)
        df['index']=index
        liste_temps=pd.DataFrame(temps)
        print()
        print("Le tableau après réduction devient: \n",df)
        return df,liste_temps
