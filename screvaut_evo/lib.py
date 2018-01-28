import numpy as np

from screvaut_evo.dat import VALID_CHARS, PS, CHAR_FREQ
from screvaut_evo import twl

from deap import tools

from tqdm import tqdm

def score(cscore, scrastr):
    string = ''.join(scrastr)
    ws = string.split(' ')

    res = sum(
            cscore[c]
            for w in ws
            for c in w
            if twl.check(w)
        )

    return res,

def draw_char():

    return np.random.choice(list(VALID_CHARS), p=PS)

def mutate(indpb, ind):

    nmut = np.random.binomial(len(ind), indpb)
    idxs = np.random.choice(len(ind), nmut)

    for idx in idxs:
        # don't change padding 0's on learning data
        if ind[idx] in CHAR_FREQ:
            ind[idx] = draw_char()

    return ind

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
