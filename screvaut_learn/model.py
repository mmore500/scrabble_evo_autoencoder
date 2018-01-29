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


        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1)


    def forward(self, x):

        x = self.cnnlayer(x)
        x = self.linlayer(x.view(x.size(0), -1))

        return x

class Model2(nn.Module):

    def __init__(self, channel_counts, kernel_sizes, input_length, dropout=None):
        super(Model2, self).__init__()

        for k in kernel_sizes:
            if k % 2 == 0:
                raise ValueError("kernel size must be odd")

        ios = zip([len(VALID_CHARS)] + channel_counts, channel_counts)

        lays = [
                lay for (i, o), k in zip(ios, kernel_sizes) for lay in (nn.Conv1d(i, o, k, padding=k//2), nn.Tanh())
            ]


        self.cnnlayer = nn.Sequential(*lays)

        self.linlayer = nn.Sequential(
                nn.Linear(input_length*channel_counts[-1], input_length*channel_counts[-1]//VALID_CHARS),
                nn.ReLU(),
                nn.Linear(input_length*channel_counts[-1]), len(VALID_CHARS))
            ) if dropout else nn.Sequential(
                nn.Linear(input_length*channel_counts[-1], input_length*channel_counts[-1]),
                nn.Tanh(),
                nn.Linear(input_length*channel_counts[-1], len(VALID_CHARS))
            )

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)


    def forward(self, x):

        x = self.cnnlayer(x)
        x = self.linlayer(x.view(x.size(0), -1))

        return x
