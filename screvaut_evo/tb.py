from deap import tools, base, creator

from screvaut_evo.lib import draw_char, score, mutate

def make_tb(p):

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    tb = base.Toolbox()

    tb.register("locus", draw_char)
    tb.register("individual", tools.initRepeat, creator.Individual, draw_char, p['indsize'])
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mutate", mutate, p['indpb'])
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("evaluate", score)

    return tb
