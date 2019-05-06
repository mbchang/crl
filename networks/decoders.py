import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, hdim, outdim, nlayers):
        super(DecoderRNN, self).__init__()
        self.hdim = hdim
        self.outdim = outdim
        self.rnn = nn.GRU(input_size=hdim, hidden_size=hdim, num_layers=nlayers, batch_first=True)
        self.fc = nn.Linear(hdim, outdim)

    def forward(self, x, hid):
        b, t, d = x.size()
        rnn_out, hid = self.rnn(x, hid)
        rnn_out = rnn_out.contiguous()
        out = self.fc(rnn_out.view(t*b,d)).view(b,t,self.outdim)
        return out, hid