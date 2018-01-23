from screvaut_evo.tb import make_tb
from screvaut_evo.dat import STDPARAM
from screvaut_evo.lib import evorun

p = STDPARAM
p['ngen'] = 10000
tb = make_tb(p)

res = evorun(tb, p)

print(res['hof'])
