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

for filename in tqdm(glob.glob('*-robustness.json')):

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


dictlist = zip(*[rep for tr in res for rep in tr])

dicts = [d for tr in res for rep in tr for tup in tr for d in tup]

indirect_rel_fits = [d['indirect_rel_fit'] for d in dicts]
indirect_dists = [d['indirect_dist'] for d in dicts]

direct_rel_fits = [d['direct_rel_fit'] for d in dicts]
direct_dists = [d['direct_dist'] for d in dicts]


directdat = list(zip(direct_dists, direct_rel_fits))
indirectdat = list(zip(indirect_dists, direct_rel_fits))

print(directdat[0:20])
print(indirectdat[0:20])

vmax1 = np.max(plot_es(directdat, "foobar", xlim, ylim, 10000))
vmax2 = np.max(plot_es(indirectdat, "foobar", xlim, ylim, 10000))

vmax = max(vmax1, vmax2)

plot_es(directdat, "Direct Encoding Evolvability Signature", xlim, ylim, vmax)

plot_es(indirectdat, "Indirect Encoding Evolvability Signature", xlim, ylim, vmax)
