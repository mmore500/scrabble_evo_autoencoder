from torch import nn

from screvaut_evo.dat import VALID_CHARS

class Model1(nn.Module):

    def __init__(self, channel_counts, kernel_sizes, input_length, dropout=None):
        super(Model1, self).__init__()

        for k in kernel_sizes:
            if k % 2 == 0:
                raise ValueError("kernel size must be odd")

        ios = zip([len(VALID_CHARS)] + channel_counts, channel_counts)

        lays = [
                lay for (i, o), k in zip(ios, kernel_sizes) for lay in ((nn.Conv1d(i, o, k, padding=k//2), nn.Tanh(), nn.Droput(p=dropout)) if dropout else (nn.Conv1d(i, o, k, padding=k//2), nn.Tanh()))
            ]


        self.cnnlayer = nn.Sequential(*lays)

        self.linlayer = nn.Sequential(
                nn.Linear(input_length*channel_counts[-1], input_length*channel_counts[-1]),
                nn.Tanh(),
                nn.Linear(input_length*channel_counts[-1], len(VALID_CHARS))
            )


    def forward(self, x):

        x = self.cnnlayer(x)
        x = self.linlayer(x.view(x.size(0), -1))

        return x
