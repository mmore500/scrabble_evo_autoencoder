import json
import glob
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

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
indirect_dists = [d['indirect_dist'] for d in dicts]

direct_rel_fits = [d['direct_rel_fit'] for d in dicts]
direct_dists = [d['direct_dist'] for d in dicts]


directdat = list(zip(direct_dists, direct_rel_fits))
indirectdat = list(zip(indirect_dists, direct_rel_fits))

vmax1 = np.max(plot_es(directdat, "foobar", xlim, ylim, 10000))
vmax2 = np.max(plot_es(indirectdat, "foobar", xlim, ylim, 10000))

vmax = max(vmax1, vmax2)

plot_es(directdat, "Direct Encoding Evolvability Signature", xlim, ylim, vmax)

plot_es(indirectdat, "Indirect Encoding Evolvability Signature", xlim, ylim, vmax)

# plot fitness versus mutational step
indirect_rel_fits_by_step = [[d['indirect_rel_fit'] for d in t] for t in dictlist]
indirect_rel_fit_mean_by_step = [np.mean(t) for t in indirect_rel_fits_by_step]
indirect_rel_fit_std_by_step = np.array([np.std(t) for t in indirect_rel_fits_by_step])

direct_rel_fits_by_step = [[d['direct_rel_fit'] for d in t] for t in dictlist]
direct_rel_fit_mean_by_step = [np.mean(t) for t in direct_rel_fits_by_step]
direct_rel_fit_std_by_step = np.array([np.std(t) for t in direct_rel_fits_by_step])

fig, ax1 = plt.subplots()
line1 = ax1.errorbar(
        range(len(indirect_rel_fit_mean_by_step)),
        indirect_rel_fit_mean_by_step,
        yerr=indirect_rel_fit_std_by_step*2/np.sqrt(len(filenames)),
        c='b',
        label="Indirect Encoding"
    )
line2 = ax1.errorbar(range(len(direct_rel_fit_mean_by_step)),
        direct_rel_fit_mean_by_step,
        yerr=direct_rel_fit_std_by_step*2/np.sqrt(len(filenames)),
        c='r',
        label="Direct Encoding"
    )
ax1.set_ylabel("Relative Fitness")
ax1.set_xlabel("Mutational Step")

ax1.legend(handles=[line1, line2])

plt.show()


# plot distance versus mutational step
indirect_dists_by_step = [[d['indirect_dist'] for d in t] for t in dictlist]
indirect_dist_mean_by_step = [np.mean(t) for t in indirect_dists_by_step]
indirect_dist_std_by_step = np.array([np.std(t) for t in indirect_dists_by_step])

direct_dists_by_step = [[d['direct_dist'] for d in t] for t in dictlist]
direct_dist_mean_by_step = [np.mean(t) for t in direct_dists_by_step]
direct_dist_std_by_step = np.array([np.std(t) for t in direct_dists_by_step])

fig, ax1 = plt.subplots()
line1 = ax1.errorbar(
        range(len(indirect_dist_mean_by_step)),
        indirect_dist_mean_by_step,
        yerr=indirect_dist_std_by_step*2/np.sqrt(len(filenames)),
        c='b',
        label="Indirect Encoding"
    )
line2 = ax1.errorbar(
        range(len(direct_dist_mean_by_step)),
        direct_dist_mean_by_step,
        yerr=direct_dist_std_by_step*2/np.sqrt(len(filenames)),
        c='r',
        label="Direct Encoding"
    )

ax1.set_ylabel("Phenotypic Distance")
ax1.set_xlabel("Mutational Step")

ax1.legend(handles=[line1, line2])

plt.show()
