from screvaut_evo.lib import mutate

import sys
from tqdm import tqdm
import json

res = None

# per-site mutation probability as first argument
indpb = float(sys.argv[1])

# number of repeated uses of a answer with different questions_answers
nrep = int(sys.argv[2])

def mutate_middle(s):
    res = list(s)
    mut = list(s)
    res[len(res)//2] = mutate(indpb, [mut[len(mut)//2]])[0]
    return ''.join(res)

with open("answers.json", 'r') as f:

    answers = json.load(f)

    res = [
            (
                    mutate_middle(a),
                    a
                ) for a in tqdm(answers * nrep)
            ]

with open("questions_answers.json", 'w') as f:
    json.dump(res, f)
