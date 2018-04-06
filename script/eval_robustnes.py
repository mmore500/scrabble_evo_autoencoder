import json
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines

from screvaut_evo.plot import plot_es

xlim = [-120,40]
ylim = [-20,120]
res = []

filenames = glob.glob('*-robustness.json')

for filename in tqdm(filenames):

    with open(filename, 'r') as f:
        if os.stat(filename).st_size == 0:
            print("warn: empty file at " + filename)
        else:
            val = json.load(f)
            base_fit = val[0][0]['direct_fit']
            for tr in val:
                for d in tr:
                    d['indirect_rel_fit'] = d['indirect_fit'] - base_fit
                    d['direct_rel_fit'] = d['direct_fit'] - base_fit

            res.append(val)


dictlist = list(zip(*[rep for tr in res for rep in tr]))

dicts = [d for tr in res for rep in tr for tup in tr for d in tup]

indirect_rel_fits = [d['indirect_rel_fit'] for d in dicts]
indirect_dists = [d['indirect_rel_dist'] for d in dicts]

direct_rel_fits = [d['direct_rel_fit'] for d in dicts]
direct_dists = [d['direct_dist'] for d in dicts]


sns.set()

rc = sns.color_palette()[1]

cdictred = {'red':   ((0.0, rc[0],rc[0]),
                   (1.0, rc[0],rc[0])),

         'green': ((0.0, rc[1],rc[1]),
                   (1.0, rc[1],rc[1])),

         'blue': ((0.0,rc[2],rc[2]),
                   (1.0,rc[2],rc[2]))}

cmapred = LinearSegmentedColormap('Red', cdictred)

bc = sns.color_palette()[0]

cdictblue = {'red':   ((0.0, bc[0], bc[0]),
                   (1.0, bc[0], bc[0])),

         'green': ((0.0, bc[1], bc[1]),
                   (1.0, bc[1], bc[1])),

         'blue': ((0.0, bc[2], bc[2]),
                   (1.0, bc[2], bc[2]))}

cmapblue = LinearSegmentedColormap('Blue', cdictblue)

# KDE takes too long without subset
idxs = np.random.randint(len(indirect_rel_fits), size=10000)

indirect_rel_fits = np.array(indirect_rel_fits)
indirect_dists = np.array(indirect_dists)
direct_rel_fits = np.array(direct_rel_fits)
direct_dists = np.array(direct_dists)

ax = sns.kdeplot(indirect_rel_fits[idxs], indirect_dists[idxs], n_levels=5, legend=True, cmap=cmapblue)
ax = sns.kdeplot(direct_rel_fits[idxs], direct_dists[idxs], n_levels=5,linestyles=['dashed'], legend=True, cmap=cmapred)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

plt.xlabel("Fitness Difference")
plt.ylabel("Phenotypic Distance")

plt.gca().invert_yaxis()
plt.title("Evolvabilty Signature Kernel Density Estimates")

blue_line = mlines.Line2D([], [], linestyle='-', color=sns.color_palette()[0],
                          markersize=15, label='Denoiser Encoding')
red_line = mlines.Line2D([], [], linestyle='--', color=sns.color_palette()[1],
                          markersize=15, label='Direct Encoding')
plt.legend(loc=4,handles=[blue_line, red_line])


plt.show()

# plot fitness versus mutational step
def df_from_key(key, valkey):
    return pd.DataFrame.from_records([{
                'Mutational Step' : istep,
                'tr' : itr,
                'rep' : irep,
                valkey : step[key]
            }
        for itr, tr in enumerate(tqdm(res)) for irep, rep in enumerate(tr) for istep, step in enumerate(rep)])

dfi = df_from_key('indirect_rel_fit', 'Fitness Difference').groupby(['tr', 'Mutational Step'], as_index=False)['Fitness Difference'].mean()
dfi['Encoding'] = 'Denoiser'

dfd = df_from_key('direct_rel_fit', 'Fitness Difference').groupby(['tr', 'Mutational Step'], as_index=False)['Fitness Difference'].mean()
dfd['Encoding'] = 'Direct'

sns.set()
ax = sns.tsplot(data=dfi, time='Mutational Step', unit='tr', value='Fitness Difference', condition='Encoding', ci=95, linestyle='-')
ax = sns.tsplot(data=dfd, time='Mutational Step', unit='tr', value='Fitness Difference', condition='Encoding', ci=95, linestyle=':', color=sns.color_palette()[1])
ax.set_xlim(0,100)

plt.title("Fitness Under Repeated Mutation")

plt.show()


dfi = df_from_key('indirect_rel_dist', 'Phenotypic Distance').groupby(['tr', 'Mutational Step'], as_index=False)['Phenotypic Distance'].mean()
dfi['Encoding'] = 'Denoiser'

dfd = df_from_key('direct_dist', 'Phenotypic Distance').groupby(['tr', 'Mutational Step'], as_index=False)['Phenotypic Distance'].mean()
dfd['Encoding'] = 'Direct'

ax = sns.tsplot(data=dfi, time='Mutational Step', unit='tr', value='Phenotypic Distance', condition='Encoding', ci=95, linestyle='-')
ax = sns.tsplot(data=dfd, time='Mutational Step', unit='tr', value='Phenotypic Distance', condition='Encoding', ci=95, linestyle=':', color=sns.color_palette()[1])
ax.set_xlim(0,100)

plt.title("Phenotypic Divergence Under Repeated Mutation")

plt.show()
