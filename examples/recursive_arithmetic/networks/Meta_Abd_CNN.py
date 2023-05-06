import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv_net(out_dim):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, out_dim),
        nn.Softmax(dim=1)
    )


class Net(nn.Module):
    def __init__(self, out_dim):
        super(Net, self).__init__()
        self.enc = conv_net(out_dim)

    def forward(self, x):
        return self.enc(x)
