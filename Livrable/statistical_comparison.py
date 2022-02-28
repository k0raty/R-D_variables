#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:29:39 2022

@author: k0raty

Se produit après tuning_batch.py , c'est un t-test.
The null hypothesis in this test is that there is no difference between the performance of two applied ML models. 
In other words, the null hypothesis assumes that both ML models perform the same. 
On the other hand, the alternative hypothesis assumes that two applied ML models perform differently.
Plus d'info sur : https://medium.com/analytics-vidhya/using-the-corrected-paired-students-t-test-for-comparing-the-performance-of-machine-learning-dc6529eaa97f
"""
import pandas as pd
from itertools import combinations
from math import factorial
import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# print correlation of AUC scores across folds
import numpy as np
from scipy.stats import t

def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

def create_model(parameters):
    """
    Crée les modèles en fonction de leurs paramètres de régularisation 

    """
    trainX=config.trainX

    n_input, n_classes = trainX.shape[1], config.n_classes 
    layers,activity_regularizer,optimizer=parameters['layers'],parameters['activity_regularizer'],parameters['optimizer']
	# create model
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=n_input, activation='relu',activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
        else:
            model.add(Dense(nodes, activation='relu',activity_regularizer=activity_regularizer))
    model.add(Dense(n_classes, activation='softmax'))

	# Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def stat_comp(results_df,n_train,n_test):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#comparing-two-models-frequentist-approach    
    
    Effectue la comparaison statistique généralisée. 
    
    ENTREE: Meilleurs modèles choisis après tuning_batch.py.
    
    SORTIE : Modèles tout aussi performant que le premier. 
    """
    
    ###On récupère les score de chaque modèle ###
    model_scores = results_df.filter(regex=r"split\d*_test_score")
    model_1_scores = model_scores.iloc[0].values  # scores of the best model
    model_2_scores = model_scores.iloc[1].values  # scores of the second-best model
    
    ###Début des tests statistiques ###
    differences = model_1_scores - model_2_scores
    
    n = differences.shape[0]  # number of test sets
    df = n - 1
    

    n_comparisons = factorial(len(model_scores)) / (
        factorial(2) * factorial(len(model_scores) - 2)
    )
    pairwise_t_test = []
    model_to_keep=[results_df.index[0]]
    for model_i, model_k in combinations(range(len(model_scores)), 2):
        model_i_scores = model_scores.iloc[model_i].values
        model_k_scores = model_scores.iloc[model_k].values
        differences = model_i_scores - model_k_scores
        t_stat, p_val = compute_corrected_ttest(differences, df, n_train, n_test)
        #p_val *= n_comparisons  # implement Bonferroni correction
        # Bonferroni can output p-values higher than 1
        p_val = 1 if p_val > 1 else p_val
        pairwise_t_test.append(
            [results_df.loc[model_scores.index[model_i]]['rank_test_score'], results_df.loc[model_scores.index[model_k]]['rank_test_score'], t_stat, p_val]
        )
        if model_i == 0:
            if p_val >0.10: #Arbitraire, 10%
                model_to_keep.append(results_df.index[model_k])
    pairwise_comp_df = pd.DataFrame(
        pairwise_t_test, columns=["model_1", "model_2", "t_stat", "p_val"]
    ).round(3)
    
    ###Affichage des résultats###
    print("On garde les modèles de rangs : %s \n" %[results_df.loc[i]['rank_test_score'] for i in model_to_keep])
    print(pairwise_comp_df)
    print("Ce tableau est enregistré sous le nom de comparaison_statistique \n")
    pairwise_comp_df.to_pickle("DataFrame/Post_paramétrage/comparaison_statistique.pkl")
    pairwise_comp_df.to_excel("Excel/Post_paramétrage/comparaison_statistique.xlsx")
    return results_df.loc[model_to_keep]

