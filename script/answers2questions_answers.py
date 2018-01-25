from screvaut_evo.lib import mutate

import sys

import json

res = None

# per-site mutation probability as first argument
indpb = int(sys.argv[1])

# number of repeated uses of a answer with different questions_answers
nrep = int(sys.argv[2])

with open("answers.json", 'r') as f:

    answers = json.load(f)

    res = [
            (
                    ''.join(mutate(indpb, list(a))),
                    a
                ) for a in answers * nrep
            ]

with open("questions_answers.json", 'w') as f:
    json.dump(res, f)
