# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:22:23 2022

@author: Antony
"""


import statistics
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt 
from tqdm import tqdm
from stats import overview
"""
 Pour chaque vecteur , on effectue une épurarion des temps trop absurde 
 pour être intégré dans le modèle. Cela permet de lisser les données et d'obtenir
 une moyenne plus cohérente pour le modèle futur. Enfin , on élimine les vecteurs
 au temps trop différent de la moyenne. 
 Standardisation de la date également.
"""
pd.options.mode.chained_assignment = None  # default='warn'

def smoothing_means(sheet,liste_temps):
    """
    Elimine les temps absurdes utilisés pour chaque vecteur 
    """
    liste_temps=liste_temps.T
    print("Lissage en cours ...\n")
    for l in tqdm(range(0,len(liste_temps.columns))): 
        q75,q25 = np.percentile(liste_temps.iloc[:,l].dropna(),[75,25])
        intr_qr = q75-q25
        max = q75+(1.5*intr_qr)
        filter = (liste_temps.iloc[:,l].dropna()<=max) #ce filtre garde l'information sur les index 
        sheet['index'].iloc[l]=pd.DataFrame(sheet['index'].iloc[l])[filter].T.values.tolist()[0] #on conserve uniquement les index encore présents
        liste_temps.iloc[:,l]=liste_temps.iloc[:,l].dropna().loc[filter].dropna() #on conserve les bon index ! ainsi on peut savoir qui a été épuré
    info=[]
    for i in range(0,len(liste_temps.columns)) : #add data that need a len >2
         L=liste_temps[i].dropna()
         if len(L) >2:
             quantiles  = np.quantile(L,[0.25,0.5,0.75])
             info.append([math.sqrt(statistics.variance(L)),quantiles[0],quantiles[2]])
         else : 
             info.append(3*[0])
    ###actualisation du sheet###   
    sheet["time"]=pd.DataFrame([statistics.mean(liste_temps[i].dropna()) for i in range(0,len(liste_temps.columns))])
    sheet['min']=pd.DataFrame([liste_temps[i].dropna().min() for i in range(0,len(liste_temps.columns))])
    sheet['max']=pd.DataFrame([liste_temps[i].dropna().max() for i in range(0,len(liste_temps.columns))])
    sheet['median']=pd.DataFrame([statistics.median(liste_temps[i].dropna()) for i in range(0,len(liste_temps.columns))])
    sheet['écart-type']=pd.DataFrame([info[i][0] for i in range(0,len(info))])
    sheet['25%']=pd.DataFrame([info[i][1] for i in range(0,len(info))])
    sheet['75%']=pd.DataFrame([info[i][2] for i in range(0,len(info))])
    sheet['lenght']=pd.DataFrame([len(liste_temps.T.iloc[i].dropna()) for i in range(0,len(liste_temps.T))]) #Nombre de temps utilisé par vecteur
    return sheet
    
    #on se sépare des vecteurs basés sur un pannel de temps trop conséquent, c'est ici qliste_temps.iloc[i].dropna()ue se forme les classes
def prunning(sheet,sheet_2):
    """
    Elimine les vecteurs aux temps trop éloigné de la moyenne
    """
    print("Epuration en cours ...\n")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20,10))
    fig.suptitle('Profil des écarts inter-quartiles des temps moyenné de nos vecteurs pour chaque set')
    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
    ax1.boxplot(sheet_2['75%']-sheet_2['25%'], flierprops=red_circle) 
    ax1.set_title("Statut initial")
    ax2.boxplot(sheet['75%']-sheet['25%'], flierprops=red_circle) 
    ax2.set_title("Après lissage")
    ax2.set( xlabel='75%-25%')

    ax1.set(ylabel='tps(ms)', xlabel='75%-25%')
    q75,q25 = np.percentile(sheet['75%']-sheet['25%'],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    filter = sheet['75%']-sheet['25%']<=max
    sheet=sheet[filter] #nouveau set retrié
    ax3.boxplot(sheet['75%']-sheet['25%'], flierprops=red_circle) 
    ax3.set_title("Après épuration")
    ax3.set( xlabel='75%-25%')
    fig.tight_layout() #Pour que l'affichage soit propre
    fig.savefig("Plots/Pré-processing/Lissage_coordonnées/ecart_inter_quartile")
    ###sécurité pour créer nos classes###
    security = np.maximum(10,int(((np.quantile(sheet['75%']-sheet['25%'],0.75)//10)+1)*10)) #prendre le maximum , les classes pourraient être trop nombreuses
    return sheet,security
def smooth (sheet,liste_temps):
    """
    Main function
    """
    sheet_2=sheet.copy()
    print("Nous comptions %s éléments \n" %len(sheet))
    overview(sheet.drop('index',axis=1),"Profil initial") #Affichage du profil, se séparer de la colonne index
    sheet = smoothing_means(sheet,liste_temps)   
    sheet,security = prunning(sheet,sheet_2)
    sheet[3]=sheet[3]/365 #standardisation de la date 
    print("Nous en comptons maintenant %s soit une réduction de %d" %(len(sheet),100-len(sheet)/len(sheet_2)*100) +'%' )
    print()
    print("Voici le tableau:\n",sheet)
    print("Ainsi que la sécurité pour créer nos classes d'intervalles  [C_k-sécurité,C_k+sécurité] : %d \n" %security)
    overview(sheet.drop('index',axis=1),"/Lissage_coordonnées/Profil après nettoyage")
    return sheet,security
                                           

