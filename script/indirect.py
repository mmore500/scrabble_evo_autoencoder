from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM, VALID_CHARS
from screvaut_evo.lib import evorun, clean
from screvaut_learn.model import Model2
import torch
import json
from scoop import futures
import matplotlib.pyplot as plt
import sys


# number of generations as first argument
ngen = int(sys.argv[1])

# path to denoising model as second argument
modelpath = sys.argv[2]

# filename to save res (logbook, hof, hofphen)
# as the third argument
resfilename = sys.argv[3]


model = Model2([9000,100], [3,5], 15)
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

if __name__ == '__main__':

    p = STDPARAM
    p['ngen'] = ngen
    p['mutpb'] = 0.33
    p['indpb'] = 0.01
    p['gpmap'] = gpmap
    tb = make_tb(p)


    res = evorun(tb, p)

    # save res
    res['hof'] = [i for i in res['hof']]
    res['hofphen'] = list(gmap(res['hof']))

    with open(resfilename, 'w') as f:
        json.dump(res, f)
