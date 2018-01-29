import matplotlib.pyplot as plt

import matplotlib.colors as colors

import numpy as np

import json

def plot_es(dat, title, xlim, ylim, vmax):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14)
    plt.subplots_adjust(top=.92)
    hexbins = plt.hexbin(np.array([x[1] for x in dat]), 100-np.array([x[0] for x in dat]), gridsize=20, cmap=plt.cm.Blues, norm=colors.LogNorm(vmin=None,vmax=vmax))
    bincounts = hexbins.get_array()
    plt.xlabel("Fitness")
    plt.ylabel("Novelty")
    cb = fig.colorbar(hexbins)
    cb.set_label('$\log_{10}(N)$')
    axes = plt.gca()
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    plt.show()

    return bincounts
