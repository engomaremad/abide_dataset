#TODO: visualize data with labels 1 and 2 using TSNE
import numpy as np
import os
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
def visualize(data,labels):
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(data)
    plt.figure(figsize=(6, 5))
    colors = ['r', 'g']
    label_names =['TD','ASD']
    for i in [1,2]:
        print(i)
        plt.scatter(X_2d[(labels).ravel() == i, 0], X_2d[(labels==i).ravel(), 1], c=colors[i-1], label=label_names[i-1])
    plt.legend(prop={'size': 20})
    plt.show()

def visualize_ados(data,labels):
    tsne = TSNE(n_components=3, random_state=0)
    X_2d = tsne.fit_transform(data)
    plt.figure(figsize=(6, 5))
    colors = ['r', 'g','b']
    label_names =['Mild','Moderate','Severe']
    for i in [1,2,3]:
        print(i)
        plt.scatter(X_2d[(labels).ravel() == i, 0], X_2d[(labels==i).ravel(), 1], c=colors[i-1], label=label_names[i-1])
    plt.legend(prop={'size': 20})
    plt.show()