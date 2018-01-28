from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM, VALID_CHARS
from screvaut_evo.lib import evorun
from screvaut_learn.lib import strings2tensor, tensor2strings
from screvaut_learn.model import Model1
import torch
from torch.autograd import Variable
import json
from scoop import futures

model = Model1([3000,100,100], [3,3,3], 15)
model.load_state_dict(torch.load('model2.pt'))
model.eval()

reps = 4

view = 15

# number of characterw viewed on left/right of center character
view_one_side = (view - 1) // 2

def clean(x):

    # 0 0 a b c d e f g 0 0
    padded = ['0'] * view_one_side + x + ['0'] * view_one_side

    # 0 0 a b c ..., 0 a b c ..., a b c ..., etc.
    zlist = [padded[i:] for i in range(view)]

    # "00a", "0abc, "abc", etc.
    x = [''.join(t) for t in zip(*zlist)]

    x = [''.join(l) for l in x]

    x = model(Variable(strings2tensor(x)))

    x = [c for s in tensor2strings(x.view(-1,len(VALID_CHARS),1).data) for c in s]

    return x

def gpmap(xs):
    for __ in range(reps):
        xs = futures.map(clean, xs)
    return xs

p = STDPARAM
p['ngen'] = 40000
p['mutpb'] = 0.4
p['indpb'] = 0.01
p['gpmap'] = gpmap
tb = make_tb(p)

if __name__ == '__main__':

    res = evorun(tb, p)

    logbook = res['logbook']

    print(json.dumps(''.join(res['hof'][0])))
    print(json.dumps(''.join(gmap([res['hof'][0]])[0])))

    import matplotlib.pyplot as plt

    gen = logbook.select("gen")
    fit_maxs = logbook.select("max")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_maxs, "b-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    plt.show()
