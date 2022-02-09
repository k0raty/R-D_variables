# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:54:03 2022

@author: Antony

test pour afficher dans une même fenêtre , subfigures ne mène à rien :\
"""
import matplotlib.pyplot as plt 

x1 = [1, 2, 3]

y1 = [4, 5, 6]


x2 = [1, 3, 5]

y2 = [6, 5, 4]

fig = plt.figure(constrained_layout=True, figsize=(10, 4))
subfigs = fig.subfigures(1, 2, wspace=0.07)
axsRight = subfigs[1].subplots(3, 1)
axsLeft = subfigs[0].subplots(3, 1)

for ax in axsRight:
    ax.plot(x1, y1)
for ax in axsLeft:
    ax.plot(x2, y2)



plt.show()