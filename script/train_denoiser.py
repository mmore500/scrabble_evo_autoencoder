import torch
from torch import nn
from screvaut_learn.model import Model2
from screvaut_learn.learn import learn
from screvaut_learn.dat import DEFAULT_LEARN_PARAMS

import sys
import json

# cuda as first argument
cuda = 'true' in sys.argv[1]

print("Cuda set to %r" % cuda)

p = DEFAULT_LEARN_PARAMS
p['print_every'] = 20
p['cuda'] = cuda
p['dropout'] = None

learning_rate = 0.01

print("loading loaders")
train = torch.load("train_loader.pt")
test = torch.load("test_loader.pt")
print("done")

print("making model")
model = Model2([9000, 100], [3, 5], 15)
print("done")

print("criterion")
criterion = nn.CrossEntropyLoss()

if p['cuda']:
    print("moving model to GPU")
    model = model.cuda()
    print("moving criterion to GPU")
    criterion = criterion.cuda()

print("optimizer")
optimizer = torch.optim.Adam(model.parameters())

print("entering learn")
model, records = learn(model, train, test, criterion, optimizer, p=p)
print("leaving learn")

torch.save(model, 'model.pt')

json.dump(records, open('record.json', 'w'))
