# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:14:51 2021

@author: Antony

Ce code prend un fichier excel de données et les tri de sorte à renvoyer un dataframe
contenant toute les informations de nos clusters. Il considère que ce ne sont pas des clusters 
mais seulement des calculs d'un matroid. Il peut y avoir des valeur NaN lorsque certaines coordonnées n'existent pas 
pour certains éléments. 
"""

import openpyxl
import pandas as pd
from tqdm import tqdm
import numpy as np
def tri(content,L):
    """
    Prend une ligne du tableur excel en entrée et le renvoie sous forme de liste
    """
    begin_index=0  #ok
    next_coor=content.find('|')
    if next_coor !=-1 :
        if content[:next_coor].find('[') !=-1:
            L.append([content[1:next_coor-1]]) #ne prend pas le dernier element
            begin_index+=next_coor+1
            return tri(content[begin_index:],L) #don't forget the return here .
        else: 
            L.append([content[:next_coor]]) #ne prend pas le dernier element
            begin_index+=next_coor+1
            return tri(content[begin_index:],L)
    else:
        return L

def load(xlsx_file,pkl_file=pd.DataFrame() ):
    """
    pkl file: Ajouter le fichier excel à cet ancien fichier (optionnel)
    Main function
   
    """
    wb_obj = openpyxl.load_workbook(xlsx_file)
    sheet = wb_obj.active
    classed_content = []
    W=[] 
    print("Création du tableau en cours : ")      
    for cell in tqdm(sheet['A']):        
        content= "%s" % cell.value #transformation en char afin d'avoir une liste
        L=tri(content,[])
        classed_content.append(L)
    print("Ajout des temps \n")
    for cell in tqdm(sheet['C']):        
        W.append(cell.value)
    df_temps=pd.DataFrame(W)
    
    df=pd.DataFrame(classed_content)
    df.columns=['var']+list(np.arange(1,len(df.columns),1)) #on crée les colonnes
    
    if len(pkl_file)!=0  :
        df_1=pd.read_pickle(pkl_file)
        df_1_temps=df_1['time']
        print("L'Ancien Tableau est : \n",df_1)
        df_1=df_1.drop(['time'],axis=1) 
        df_1=pd.concat([df,df_1],axis=0) #on colle les deux tableaux
        df_1_temps=pd.concat([df_temps,df_1_temps],axis=0) #on s'occupe des temps
        df_1=pd.concat([df_1,df_1_temps],axis=1)
        df_1=df_1.drop([0],axis=0) # retire une colonne none qui sert à rien 
        df_1=df_1.rename(columns={0: 'time'})
        df_1=df_1.astype({"time":float}) #Utile de changer le type pour la suite !
        df_1= df_1.reset_index() #important pour éviter d'avoir des doublons ! On perd en effet l'indice des calculs initiaux mais cela n'a pas d'importance si on ne souhaite qu'entraîner le model
        df_1=df_1.drop(['index'],axis=1) 
        print("Le tableau devient  \n:", df_1)
        return df_1
    else:
        df=pd.concat([df,df_temps],axis=1)
        df=df.drop([0],axis=0) #une colonne none qui sert à rien 
        df=df.rename(columns={0: 'time'})
        df=df.astype({"time":float}) #it's string !
        print()
        print("Le tableau est : \n", df)
        print()
        return df
    
