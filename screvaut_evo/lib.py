import numpy as np

from screvaut_evo.dat import VALID_CHARS, PS, CHAR_FREQ, LETTER_CHARACTER_SCORE
from screvaut_evo import twl
from screvaut_learn.lib import strings2tensor, tensor2strings

from deap import tools

from tqdm import tqdm

from torch.autograd import Variable

def score(cscore, scrastr):
    string = ''.join(scrastr)
    ws = string.split(' ')

    res = sum(
            LETTER_CHARACTER_SCORE[c]
            for w in ws
            for c in w
            if twl.check(w)
        )

    return res,

def draw_char():

    return np.random.choice(list(VALID_CHARS), p=PS)

def mutate(indpb, ind):

    nmut = np.random.binomial(len(ind), indpb)

    return perform_mut(nmut, ind)

def perform_mut(nmut, ind):
    idxs = np.random.choice(len(ind), nmut)

    for idx in idxs:
        # don't change padding 0's on learning data
        if ind[idx] in CHAR_FREQ:
            ind[idx] = draw_char()

    return ind

def clean(x, model, view):

    # number of characterw viewed on left/right of center character
    view_one_side = (view - 1) // 2


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

def scrastr_phen_dist(a,b):
    return len(a) - sum(1 for c,d in zip(a,b) if c == d)


def evorun(tb, p):

    # initialize stats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # initialize logbook
    logbook = tools.Logbook()

    # initialize hall of fame
    hof = tools.HallOfFame(p['nhof'])

    # initialize population
    pop = tb.population(n=p['npop'])

    # Evaluate the entire population
    fitnesses = map(tb.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    try:
        for g in tqdm(range(p['ngen'])):
            # Select the next generation individuals
            offspring = tb.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(tb.clone, offspring))

            # Apply mutation to offspring
            for mutant in offspring:
                if np.random.random() < p['mutpb']:
                    tb.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_phen = p['gpmap'](invalid_ind)
            fitnesses = map(tb.evaluate, invalid_phen)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Record data
            hof.update(pop)
            print(hof[0].fitness.values[0])
            record = stats.compile(pop)
            logbook.record(gen=g, **record)

    except KeyboardInterrupt:
        print("interrupted")

    return {
            'pop' : pop,
            'logbook' : logbook,
            'hof' : hof
        }
