import matplotlib
import pylab as pl
from sklearn import datasets
import numpy as np
from itertools import cycle

irisData = datasets.load_iris()
print (irisData.data)
print (irisData.target)
print(irisData.feature_names)
print(irisData.target_names)

print(f"attributes of 32nd element: {irisData.data[31]} the class is: {irisData.target_names[irisData.target[31]]} ({irisData.target[31]})")

for index in np.unique(irisData.target):
    print("type " + irisData.target_names[index] + " contains "+ str(len(irisData.target[irisData.target == index])) + " instances")

def plot_2D_sepal(data, target, target_names):
    colors = cycle('rgbcmykw') # cycle de couleurs
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)  #on utilise les attributs sepal
    pl.legend()
    pl.show()


def plot_2D_petal(data, target, target_names):
    colors = cycle('rgbcmykw') # cycle de couleurs
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names): #i attribut c colors label for names
        pl.scatter(data[target == i, 2], data[target == i, 3], c=c, label=label)  #on utilise les attributs pétales
    pl.legend()
    pl.show()

def plot_2D_petal_reg(data, target, target_names):
    colors = cycle('rgbcmykw') # cycle de couleurs
    target_ids = range(len(target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, target_names): #i attribut c colors label for names
        pl.scatter(data[target == i, 2], data[target == i, 3], c=c, label=label)  #on utilise les attributs pétales
    pl.plot([2.5,2.5],[0,2.5]) 
    pl.legend()
    pl.show()


plot_2D_sepal(irisData.data, irisData.target, irisData.target_names)
plot_2D_petal(irisData.data, irisData.target, irisData.target_names)
plot_2D_petal_reg(irisData.data, irisData.target, irisData.target_names)