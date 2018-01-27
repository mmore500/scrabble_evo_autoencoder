import torch
from torch.autograd import Variable
import itertools
import time
import json

from screvaut_evo.dat import VALID_CHARS

from screvaut_learn.lib import tensor2strings, timeSince
from screvaut_learn.dat import DEFAULT_LEARN_PARAMS

def learn(model, train_loader, test_loader, criterion, optimizer, p=DEFAULT_LEARN_PARAMS, nepochs=None):

    start = time.time()

    record = list()

    # Train the Model
    model.train()
    try:

        for epoch in (nepochs or itertools.count()):
            for i, (questions, answers) in enumerate(train_loader):
                if p['cuda']:
                    questions, answers = questions.cuda(), answers.cuda()

                questions = Variable(questions).float()
                answers = Variable(answers).long()
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(questions)
                train_loss = criterion(outputs, answers)
                train_loss.backward()
                optimizer.step()

                print('.', end="")

                if p['cuda']:
                    del questions
                    del answers
                    torch.cuda.empty_cache()

                if (i+1) % p['print_every'] == 0:
                    model.eval()

                    # evaluate on test data
                    questions, answers = next(test_loader.__iter__())

                    if p['cuda']:
                        questions, answers = questions.cuda(), answers.cuda()

                    questions = Variable(questions).float()
                    answers = Variable(answers).long()
                    outputs = model(questions)
                    test_loss = criterion(outputs, answers)

                    _, predicted = torch.max(outputs.data, 1)
                    total = answers.size(0)
                    correct = (predicted == answers.data).sum()

                    print()
                    print('Epoch [%d], Iter [%d/%d], Time %s, Train Loss: %.4f, Test Loss: %.4f, Test Performance: %.4f'
                           %(epoch+1, i+1, len(train_loader)//train_loss.data.size(0), timeSince(start), train_loss.data[0], test_loss.data[0], correct/total))

                    record.append({
                        'epoch' : epoch+1,
                        'iter' : i+1,
                        'train_loss' : train_loss.data[0],
                        'test_loss' : test_loss.data[0],
                        'test_performance' : correct/total
                        })
                    json.dump(record, open('record.json', 'w'))

                    question_strings = tensor2strings(questions.data)
                    answer_strings = [str(VALID_CHARS[int(a.data[0])]) for a in answers]
                    guess_strings = tensor2strings(outputs.view(-1,len(VALID_CHARS),1).data)

                    for q, a, g in itertools.islice(zip(question_strings, answer_strings, guess_strings), p['examples_per']):
                        mark = '✓' if g == a else '✗ (%s)' % a
                        print('=' * 30)
                        print(' ' * (len(q)//2+10) +  '|')

                        print('question: %s' % q)
                        print('guess: %s %s' % (g, mark))

                    if p['cuda']:
                        del questions
                        del answers
                        torch.cuda.empty_cache()

                    model.train()


                if (i+1) % p['checkpoint_every'] == 0:
                    print('Checkpointing model...')
                    torch.save(model, 'model.pt')

    except KeyboardInterrupt:
        print('Checkpointing model...')
        torch.save(model, 'model.pt')

    return model, record
