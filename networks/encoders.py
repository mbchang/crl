import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, activation=None, predictor=None):
        super(Identity, self).__init__()
        self.activation = activation
        self.predictor = predictor

    def forward(self, x):
        if self.activation is not None:
            x = self.activation(x)
        if self.predictor is not None:
            pred = self.predictor(x)
            return x, pred
        else:
            return x

    def get_parameter(self):
        return ''

class EncoderRNN(nn.Module):
    def __init__(self, indim, hdim, nlayers):
        super(EncoderRNN, self).__init__()
        self.indim = indim
        self.hdim = hdim
        self.rnn = nn.GRU(input_size=indim, hidden_size=hdim, num_layers=nlayers, batch_first=True)

    def forward(self, x, hid):
        rnn_out, hid = self.rnn(x, hid)
        rnn_out = F.relu(rnn_out)
        return rnn_out, hid

class CNN64fc_8(nn.Module):
    """ Total Parameters: 193216 """
    def __init__(self, outdim, activation=None, predictor=None):
        super(CNN64fc_8, self).__init__()
        self.outdim = outdim
        self.activation = activation
        self.predictor = predictor

        nc = 1
        ndf = 8
        self.network = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, self.outdim, 4, 1, 0, bias=False))

    def forward(self, x):
        bsize = x.size(0)
        x = self.network(x)  # (bsize, num_outdim, 1, 1)
        x = x.view(bsize, self.outdim)
        if self.activation is not None:
            x = self.activation(x)
        if self.predictor is not None:
            pred = self.predictor(x)
            return x, pred
        else:
            return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

