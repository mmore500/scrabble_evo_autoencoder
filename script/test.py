from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM
from screvaut_evo.lib import evorun

import os

p = STDPARAM
p['ngen'] = 500
tb = make_tb(p)

res = evorun(tb, p)

logbook = res['logbook']

print(res['hof'][0])
print(res['hof'][0].fitness.values)

gen = logbook.select("gen")
fit_maxs = logbook.select("max")

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, fit_maxs, "b-", label="Maximum Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

plt.show()
