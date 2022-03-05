# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:57:18 2022

@author: Antony
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config 
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold 
from silence_tensorflow import silence_tensorflow
import pandas as pd
silence_tensorflow()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# fit a model and plot learning curve
def create_model(parameters):
    """
    Crée les modèles en fonction de leur paramètres de régularisation 

    """
    trainX=config.trainX
    n_input, n_classes = trainX.shape[1], config.n_classes
    layers,activation,optimizer,activity_regularizer=parameters['parameters']['layers'],parameters['parameters']['activation'],parameters['parameters']['optimizer'],parameters['activity_regularizer']
	# create model
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=n_input, activation=activation,activity_regularizer=activity_regularizer, kernel_initializer='he_uniform'))
        else:
            model.add(Dense(nodes, activation=activation,activity_regularizer=activity_regularizer))
    model.add(Dense(n_classes, activation='softmax'))

	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def model_display(results):
    """
    Main function 
    """
    results=results[:2]
    trainX, trainy=config.trainX, config.trainy
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    cv = KFold(n_splits=5,shuffle=True) #Cette cross-validation est obligatoire pour le t-test

    ###Premier modèle###
    print("Etude du premier modèle\n")
    model = KerasClassifier(build_fn=create_model,epochs=300,batch_size=results.iloc[0]['param_batch_size'],verbose=0)
    param_grid = dict(parameters=[results.iloc[0]['param_parameters']])
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv,verbose=0)
    title = "Learning Curves of the first model"
    plot_learning_curve(
        grid, title, trainX, trainy, axes=axes[:, 0], cv=cv, n_jobs=-1
    )
    if len(results)>1:
        ###Second modèle###
        print("Etude du second modèle \n")
        model = KerasClassifier(build_fn=create_model,epochs=300,batch_size=results.iloc[1]['param_batch_size'],verbose=0)
        param_grid = dict(parameters=[results.iloc[1]['param_parameters']])
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv,verbose=0)
        title = "Learning Curves of the second model"
        plot_learning_curve(
            grid, title, trainX, trainy, axes=axes[:, 1], cv=cv, n_jobs=-1
        )
    fig.savefig("Plots/Post_paramétrage/Courbes_d'apprentissage")
    print("Profil image enregistré sous le nom de Courbes_d'apprentissage")

