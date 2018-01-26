import torch
import time
import math
from screvaut_evo.dat import CHAR_IDX, VALID_CHARS

# source:
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Turn a string into a <len(strings) x len(CHAR_IDX) x len(string)>,
# or an array of one-hot letter vectors
def strings2tensor(strings):

    maxstringlen = max(len(string) for string in strings)
    tensor = torch.zeros(len(strings), len(CHAR_IDX), maxstringlen)

    for batch_idx, string in enumerate(strings):
        for string_idx, c in enumerate(string):
            # exclude 0 buffer character
            if c in CHAR_IDX:
                tensor[batch_idx][CHAR_IDX[c]][string_idx] = 1

    return tensor

def tensor2strings(tensor):

    strings = []

    vals, idxss = tensor.topk(1, dim=1)

    strings = [''.join(
                    VALID_CHARS[idx] for idx in idxs
                ) for idxs, in idxss.tolist()]

    return strings

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
