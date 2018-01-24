import json
import glob
from tqdm import tqdm
import os

# consolodate all json files containing champion individuals from direct
# evolutionary runs into a single json file

scrastrs = []

for filename in tqdm(glob.glob('*')):

    with open(filename, 'r') as f:
        if os.stat(filename).st_size == 0:
            print("warn: empty file at " + filename)
        else:
            scrastr = json.load(f)
            if scrastr and isinstance(scrastr, str):
                scrastrs.append(scrastr)
            else:
                print("warn: invalid scrastr for " + filename)


with open("direct_champions.json", 'w') as f:
    json.dump(scrastrs, f)
