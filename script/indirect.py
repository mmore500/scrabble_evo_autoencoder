from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM, VALID_CHARS
from screvaut_evo.lib import evorun, clean
from screvaut_learn.model import Model1
import torch
import json
from scoop import futures
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

    modelpath = sys.argv[1]

    model = Model1([3000,100,100], [3,3,3], 15)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    reps = 4

    view = 15

    def myclean(x):
        return clean(x, model, view)

    def gpmap(xs):
        for __ in range(reps):
            xs = futures.map(myclean, xs)
        return xs

    p = STDPARAM
    p['ngen'] = 40000
    p['mutpb'] = 0.4
    p['indpb'] = 0.01
    p['gpmap'] = gpmap
    tb = make_tb(p)

    res = evorun(tb, p)

    logbook = res['logbook']

    print(json.dumps(''.join(res['hof'][0])))
    print(json.dumps(''.join(gmap([res['hof'][0]])[0])))

    gen = logbook.select("gen")
    fit_maxs = logbook.select("max")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_maxs, "b-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    plt.show()
