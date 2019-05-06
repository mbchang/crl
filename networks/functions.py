import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import cuda_if_needed, reverse, sample_from_categorical_dist, logprob_categorical_dist
from torch.autograd import Variable

import numpy as np

from encoders import CNN64fc_8

class Function(nn.Module):
    def __init__(self, indim, outdim):
        super(Function, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.fc = nn.Linear(indim, outdim)

    def forward(self, x):
        return F.relu(self.fc(x))

class TransformFixedLength(nn.Module):
    def __init__(self, indim, hdim, outdim, nlayers, args):
        super(TransformFixedLength, self).__init__()
        self.args = args
        hdim -= (hdim % 3)  # round down to nearest multiple of 3
        self.hdim = hdim
        self.enc = nn.Linear(indim, hdim//3)
        self.layers = nn.ModuleList([nn.Linear(hdim, hdim) for i in range(nlayers)])
        self.dec = nn.Linear(hdim, outdim)

    def forward(self, x, y=None, mode=None):
        b,t,d = x.size()
        assert t == 3
        # use the same weights for each entry along the time dimension
        h = self.enc(x.view(b*t, d))
        # concatenate out the time dimension
        h = h.view(b, self.hdim)  
        h = F.relu(h)
        for layer in self.layers:
            h = F.relu(layer(h))
        o = self.dec(h)  # (b, outdim)
        return o, F.softmax(o, dim=1)

class OperatorFixedLength(nn.Module):
    def __init__(self, indim, hdim, outdim, nlayers, args):
        super(OperatorFixedLength, self).__init__()
        self.args = args
        assert hdim % 2 == 0
        self.enc1 = nn.Linear(indim, hdim//2)
        self.enc2 = nn.Linear(indim, hdim//2)
        self.layers = nn.ModuleList([nn.Linear(hdim, hdim) for i in range(nlayers)])
        self.dec = nn.Linear(hdim, outdim)

    def forward(self, x):
        x1, x2 = x
        x1, x2 = torch.squeeze(x1, dim=1), torch.squeeze(x2, dim=1)
        h1, h2 = F.relu(self.enc1(x1)), F.relu(self.enc2(x2))
        h = torch.cat((h1, h2), dim=1)
        for layer in self.layers:
            h = F.relu(layer(h))
        o = self.dec(h)
        return o, F.softmax(o, dim=1)

class EncoderDecoderRNN(nn.Module):
    def __init__(self, indim, hdim, outdim, outlength, nlayers, args):
        super(EncoderDecoderRNN, self).__init__()
        self.args = args
        self.indim = indim
        self.hdim = hdim
        self.outdim = outdim
        self.outlength = outlength
        self.nlayers = nlayers
        self.encoder = EncoderRNN(2*indim, hdim, nlayers)
        self.decoder = DecoderRNN(hdim, outdim, nlayers)

    def init_hidden(self, bsize):
        assert self.args.bidirectional == False
        (num_directions, h_hdim) = (2, self.hdim//2) if self.args.bidirectional else (1, self.hdim)
        return cuda_if_needed(Variable(torch.zeros(self.nlayers*num_directions, bsize, h_hdim)), self.args)

    def pad(self, encoder_out):
        b, d = encoder_out.size()
        encoder_out = encoder_out.unsqueeze(1)  # unsqueeze the time dimension
        padding = cuda_if_needed(Variable(torch.zeros(b, self.outlength-1, d)), self.args)
        padded_encoder_out = torch.cat((encoder_out, padding), dim=1)
        padded_encoder_out = padded_encoder_out.contiguous()
        return padded_encoder_out

    def forward(self, x):
        x1, x2 = x
        # pad to outlength-1
        # padding is added the highest digit
        # ones digit, ..., last digit, <padding>
        assert x1.size(1) == self.outlength
        assert x2.size(1) == self.outlength

        # # now reverse, such that the ones digit is seen last
        # now concatenate
        x = torch.cat((x1, x2), dim=2)
        # now input to network
        b, t, d = x.size()
        rnn_hid = self.init_hidden(b)  # (nlayers, b, h)
        encoder_out, encoder_hid = self.encoder(x, rnn_hid)  # encoder_out (b, t, h), encoder_hid (nlayers, b, h)
        padded_encoder_out = self.pad(encoder_out[:, -1])  # padded_encoder_out (b, outlength, h)
        decoder_out, _ = self.decoder(padded_encoder_out, encoder_hid)  # decoder_out (bsize, outlength, outdim)
        decoder_out = reverse(decoder_out, dim=1)
        return decoder_out, F.softmax(decoder_out, dim=-1)

class PlainTranslator(nn.Module):
    def __init__(self, indim, args):
        super(PlainTranslator, self).__init__()
        self.args = args
        self.indim = indim
        self.fc = nn.Linear(indim, indim)

    def forward(self, x, y=None, mode=None):
        return self.fc(x)

class Localization64(nn.Module):
    def __init__(self):
        super(Localization64, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        nc = 1
        ndf = 8
        self.localization = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 8 x 8
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf) x 4 x 4
            )

    def forward(self, x):
        return self.localization(x)

class FCLoc(nn.Module):
    def __init__(self, indim, outdim):
        super(FCLoc, self).__init__()
        self.fc_loc = nn.Sequential(
            nn.Linear(indim, 32),
            nn.ReLU(True),
            nn.Linear(32, outdim)
        )

    def forward(self, x):
        return self.fc_loc(x)

    def get_parameter(self):
        """ gets the parameter from the last forward pass """
        return self.p

    def set_parameter(self, p):
        self.p = p

class AffineFCLoc(FCLoc):
    def __init__(self, indim):
        super(AffineFCLoc, self).__init__(indim, outdim=3*2)
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, x):
        """
        [[a, b, c],
         [d, e, f]]
        """
        theta = super(AffineFCLoc, self).forward(x).view(-1, 2, 3)
        return theta

class TranslateFCLoc(FCLoc):
    def __init__(self, indim):
        super(TranslateFCLoc, self).__init__(indim, outdim=2)
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([0,0])

    def get_parameter(self):
        return 'h: {} v: {}'.format(self.p[1], self.p[0])

    def forward(self, x):
        """
        [[1, 0, x],
         [0, 1, y]]
        """
        theta_translation = super(TranslateFCLoc, self).forward(x).unsqueeze(2)  # (bsize, 2, 1)
        self.set_parameter(theta_translation[-1].squeeze().data.cpu().numpy())  # (2,) numpy array
        theta = torch.zeros(x.size(0), 2, 2)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta = Variable(theta, requires_grad=False)
        if x.is_cuda:
            theta = theta.cuda()
        theta = torch.cat([theta, theta_translation], dim=2)
        return theta

class RotateFCLoc(FCLoc):
    def __init__(self, indim):
        super(RotateFCLoc, self).__init__(indim, outdim=2*2)
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1,0,0,1])

    def forward(self,x):
        """
        [[cos(), sin(), 0],
         [-sin(), cos(), 0]]
        """
        theta_rotation = super(RotateFCLoc, self).forward(x).view(-1, 2, 2)  # (bsize, 2, 2)
        theta = torch.zeros(x.size(0), 2, 1)
        theta = Variable(theta, requires_grad=False)
        if x.is_cuda:
            theta = theta.cuda()
        theta = torch.cat([theta_rotation, theta], dim=2)
        return theta

class ConstrainedRotateFCLoc(FCLoc):
    def __init__(self, indim):
        super(ConstrainedRotateFCLoc, self).__init__(indim, outdim=2*1)
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1,0])

    def get_parameter(self):
        return 'cos: {} sin: {}'.format(self.p[0], self.p[1])

    def forward(self, x):
        """
        [[cos(), sin(), 0],
         [-sin(), cos(), 0]]
        """
        theta_rotation_top = super(ConstrainedRotateFCLoc, self).forward(x).view(-1, 1, 2)  # (bsize, 1, 2)
        self.set_parameter(theta_rotation_top[-1].squeeze().data.cpu().numpy())
        cos = theta_rotation_top[:, :, 0]
        sin = theta_rotation_top[:, :, 1]

        theta_rotation_bottom = torch.stack((-sin, cos), dim=-1)
        theta_rotation = torch.cat((theta_rotation_top, theta_rotation_bottom), dim=1)

        theta = torch.zeros(x.size(0), 2, 1)
        theta = Variable(theta, requires_grad=False)
        if x.is_cuda:
            theta = theta.cuda()
        theta = torch.cat([theta_rotation, theta], dim=2)
        return theta

class ScaleFCLoc(FCLoc):
    def __init__(self, indim):
        super(ScaleFCLoc, self).__init__(indim, outdim=2*1)
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1,1])

    def get_parameter(self):
        return 'x: {} y: {}'.format(self.p[0], self.p[1])

    def forward(self,x):
        """
        [[x, 0, 0],
         [0, y, 0]]
        """
        theta_scale = super(ScaleFCLoc, self).forward(x).unsqueeze(2)  # (bsize, 2, 1)
        self.set_parameter(theta_scale[-1].squeeze().data.cpu().numpy())
        theta = torch.zeros(x.size(0), 2, 2)
        theta = Variable(theta, requires_grad=False)
        if x.is_cuda:
            theta = theta.cuda()

        theta = torch.cat([theta_scale, theta], dim=2)

        theta_top = theta[:, 0].unsqueeze(1)
        theta_bottom = theta[:, 1]
        permute = torch.LongTensor([2, 0, 1])
        if x.is_cuda:
            permute = permute.cuda()
        theta_bottom = theta_bottom[:, permute].unsqueeze(1)

        theta = torch.cat([theta_top, theta_bottom], dim=1)

        return theta

class STNTransformation(nn.Module):
    def __init__(self, indim, fcloc):
        super(STNTransformation, self).__init__()
        if indim == (64, 64):
            self.hdim = 8*4*4
            self.localization = Localization64()
        else:
            assert False
        self.fc_loc = fcloc(indim=self.hdim)

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.hdim)
        theta = self.fc_loc(xs)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode='border')
        return x

    def get_parameter(self):
        """ gets the parameter from the last forward pass """
        return self.fc_loc.get_parameter()

class AffineSTNTransformation(STNTransformation):
    def __init__(self, indim, outdim):
        super(AffineSTNTransformation, self).__init__(
            indim=indim,
            fcloc=AffineFCLoc)

class TranslateSTN(STNTransformation):
    def __init__(self, indim, outdim):
        super(TranslateSTN, self).__init__(
            indim=indim,
            fcloc=TranslateFCLoc)

class RotateSTN(STNTransformation):
    def __init__(self, indim, outdim):
        super(RotateSTN, self).__init__(
            indim=indim,
            fcloc=RotateFCLoc)

class ConstrainedRotateSTN(STNTransformation):
    def __init__(self, indim, outdim):
        super(ConstrainedRotateSTN, self).__init__(
            indim=indim,
            fcloc=ConstrainedRotateFCLoc)

class ScaleSTN(STNTransformation):
    def __init__(self, indim, outdim):
        super(ScaleSTN, self).__init__(
            indim=indim,
            fcloc=ScaleFCLoc)
