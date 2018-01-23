from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM
from screvaut_evo.lib import evorun

import json

p = STDPARAM
p['ngen'] = 40000
tb = make_tb(p)

res = evorun(tb, p)

logbook = res['logbook']

print(json.dumps(''.join(res['hof'][0])))