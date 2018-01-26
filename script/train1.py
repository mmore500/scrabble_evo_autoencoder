import torch
from torch import nn
from screvaut_learn.model import Model1
from screvaut_learn.learn import learn
from screvaut_learn.dat import DEFAULT_LEARN_PARAMS

p = DEFAULT_LEARN_PARAMS
p['print_every'] = 10

learning_rate = 0.01

train = torch.load("train_loader.pt")
test = torch.load("test_loader.pt")

model = Model1([2000,1000],[3,3], 15)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

learn(model, train, test, criterion, optimizer, p=p)
