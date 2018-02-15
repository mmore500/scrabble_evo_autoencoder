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


direct_filenames = glob.glob('direct_res/*-res.json')
indirect_filenames = glob.glob('indirect_res/*-res.json')

pages = list()

for i, filename in enumerate(tqdm(direct_filenames)):

    with open(filename, 'r') as f:
        val = json.load(f)

        for p in val['logbook']:
            p['rep'] = i
            p['Generation'] = p['gen']
            p['Maximum Fitness'] = p['max']
            p['Encoding'] = 'Direct'

        pages.extend([p for p in val['logbook']])

dfdirect = pd.DataFrame.from_records(pages)

pages = list()

for i, filename in enumerate(tqdm(indirect_filenames)):

    with open(filename, 'r') as f:
        val = json.load(f)

        for p in val['logbook']:
            p['rep'] = i
            p['Generation'] = p['gen']
            p['Maximum Fitness'] = p['max']
            p['Encoding'] = 'Denoiser'

        pages.extend([p for p in val['logbook']])


dfindirect = pd.DataFrame.from_records(pages)


sns.set()
ax = sns.tsplot(data=dfindirect, time='Generation', unit='rep', condition='Encoding', value='Maximum Fitness', ci=95, linestyle='-')
ax = sns.tsplot(data=dfdirect, time='Generation', unit='rep', condition='Encoding', value='Maximum Fitness', ci=95, linestyle=':', color=sns.color_palette()[1])

plt.title("Maximum Fitness by Generation")

plt.show()
