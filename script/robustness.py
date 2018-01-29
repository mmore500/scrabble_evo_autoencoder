from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM, VALID_CHARS, LETTER_CHARACTER_SCORE
from screvaut_evo.lib import score, perform_mut, clean, scrastr_phen_dist
from screvaut_learn.lib import strings2tensor, tensor2strings
from screvaut_learn.model import Model1, Model2
from tqdm import tqdm
import torch
from torch.autograd import Variable
import json
from scoop import futures
import sys
import os

clean_reps = 4
mut_run_reps = 50
num_muts = 100
view = 15

# target json filename as first argument
json_filename = sys.argv[1]

# model filename as second argument
model_filename = sys.argv[2]

# out filename as third argument
out_filename = sys.argv[3]

# settings shouldn't matter to load from file
# lies: this does matter
model = Model2([9000,100], [3,5], 15)
model.load_state_dict(torch.load(model_filename))
model.eval()

scrastr = None

with open(json_filename, 'r') as f:
    if os.stat(json_filename).st_size == 0:
        print("fail: empty file at " + json_filename)
    else:
        scrastr = json.load(f)
        if scrastr and isinstance(scrastr, str):
            pass
        else:
            print("fail: invalid scrastr for " + json_filename)


def myclean(x):
    return clean(x, model, view)

def gpmap(x):
    for __ in range(clean_reps):
        x = myclean(x)
    return x

def make_run_res(__):
    run_res = []

    champ = [c for c in scrastr]
    originalindirectphen = list(gpmap(curchamp))

    curchamp = [c for c in scrastr]

    for __ in tqdm(range(num_muts)):

        indirect_phen = list(gpmap(curchamp))
        direct_phen = curchamp

        rdict = {}

        rdict['indirect_fit'] = score(LETTER_CHARACTER_SCORE ,indirect_phen)[0]
        rdict['direct_fit'] = score(LETTER_CHARACTER_SCORE, direct_phen)[0]
        rdict['indirect_dist'] = scrastr_phen_dist(champ, indirect_phen)
        rdict['indirect_rel_dist'] = scrastr_phen_dist(originalindirectphen, indirect_phen)
        rdict['direct_dist'] = scrastr_phen_dist(champ, direct_phen)

        curchamp = perform_mut(1, curchamp)

        run_res.append(rdict)

    return run_res

if __name__ == '__main__':

    res = []

    res = list(futures.map(make_run_res, range(mut_run_reps)))

    print(res)

    with open(out_filename, 'w') as f:
        json.dump(res, f)
