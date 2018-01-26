from torch import nn

from screvaut_evo.dat import VALID_CHARS

class Model1(nn.Module):

    def __init__(self, channel_counts, kernel_sizes, input_length, dropout=None):
        super(Model1, self).__init__()

        for k in kernel_sizes:
            if k % 2 == 0:
                raise ValueError("kernel size must be odd")

        ios = zip([len(VALID_CHARS)] + channel_counts, channel_counts)

        self.cnnlayers = [
            nn.Sequential(
                nn.Conv1d(i, o, k, padding=k//2),
                nn.Tanh(),
            ) for (i, o), k in zip(ios, kernel_sizes)]

        self.linlayer = nn.Sequential(
                nn.Linear(input_length*channel_counts[-1], input_length*channel_counts[-1]),
                nn.Tanh(),
                nn.Linear(input_length*channel_counts[-1], len(VALID_CHARS))
            )

        self.dodropout = dropout
        self.dropouts = [nn.Dropout(p=(dropout or 0) for __ in self.cnnlayers]

    def forward(self, x):

        for lay, d in zip(self.cnnlayers, self.dropouts):
            x = lay(x)
            if self.dodropout:
                x = d(x)

        x = self.linlayer(x.view(x.size(0), -1))

        return x
