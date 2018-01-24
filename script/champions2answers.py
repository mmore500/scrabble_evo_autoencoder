import json
from tqdm import tqdm
import sys

# number of characters viewed by CNN (first argument)
view = int(sys.argv[1])

# number of characterw viewed on left/right of center character
view_one_side = (view - 1) // 2

answers = []

with open("direct_champions.json", 'r') as f:
    scrastrs = json.load(f)

    for s in tqdm(scrastrs):

        # 0 0 a b c d e f g 0 0
        padded = '0' * view_one_side + s + '0' * view_one_side

        # 0 0 a b c ..., 0 a b c ..., a b c ..., etc.
        zlist = [padded[i:] for i in range(view)]

        # "00a", "0abc, "abc", etc.
        answers.extend(''.join(t) for t in zip(*zlist))

with open("answers.json", 'w') as f:
    json.dump(answers, f)
