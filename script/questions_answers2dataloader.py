import torch
import torch.utils.data as data_utils

from screvaut_learn.lib import strings2tensor
from screvaut_evo.dat import CHAR_IDX
from tqdm import tqdm

import sys
import json

# batch sizes as first argument
bsize = int(sys.argv[1])

questions_answers = None

with open("questions_answers.json", 'r') as f:
    questions_answers = json.load(f)


features = strings2tensor([q for q, __ in tqdm(questions_answers)])

# we just want the net to predict the middle char of the string
targets = torch.IntTensor([CHAR_IDX[a[len(a)//2]] for __, a in tqdm(questions_answers)])

# source:
# https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets
train = data_utils.TensorDataset(features, targets)
train_loader = data_utils.DataLoader(train, batch_size=bsize, shuffle=True)

torch.save(train_loader, "loader.pt")
