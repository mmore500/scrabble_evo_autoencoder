from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM
from screvaut_evo.lib import evorun

import sys
import json

# number of generations as first argument
ngen = int(sys.argv[1])

# filename to save res (logbook, hof)
# optional
resfilename = sys.argv[2] if len(sys.argv) > 2 else None

p = STDPARAM
p['ngen'] = ngen
p['mutpb'] = 0.33
p['indpb'] = 0.01

tb = make_tb(p)

res = evorun(tb, p)

# keep this part for use of direct to create training data
# accomplished by redirecting stdout to a file
print(json.dumps(''.join(res['hof'][0])))

# save res
res['hof'] = [i for i in res['hof']]
res['hofphen'] = [i for i in res['hof']]

if resfilename:
    with open(resfilename, 'w') as f:
        json.dump(res, f)
