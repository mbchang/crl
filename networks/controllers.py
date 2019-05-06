import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import cuda_if_needed, reverse, sample_from_categorical_dist, logprob_categorical_dist
from torch.autograd import Variable

from encoders import CNN64fc_8

class BasePolicy(nn.Module):
    def __init__(self):
        super(BasePolicy, self).__init__()
        self.net = None

    def forward(self, state):
        action_dist = F.softmax(self.net(state), dim=1)
        return action_dist

    def select_action(self, state):
        # volatile
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        action = m.sample()
        return action.data

    def get_log_prob(self, state, action):
        # not volatile
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        log_prob = m.log_prob(action)
        return log_prob

class BaseValueFn(nn.Module):
    def __init__(self):
        super(BaseValueFn, self).__init__()
        self.net = None

    def forward(self, state):
        value = self.net(state)
        return value

class CNNPolicy(BasePolicy):
    def __init__(self, indim, num_actions):
        super(CNNPolicy, self).__init__()

        if indim == (64, 64):
            self.net = CNN64fc_8(num_actions)  # (indim, num_actions)
        else:
            assert False

class CNNValueFn(BaseValueFn):
    def __init__(self, indim):
        super(CNNValueFn, self).__init__()

        if indim == (64, 64):
            self.net = CNN64fc_8(1)  # (indim, num_actions)
        else:
            assert False

class GenericRNN(nn.Module):
    def __init__(self, indim, hdim, nlayers, args):
        super(GenericRNN, self).__init__()
        self.args = args
        self.indim = indim
        self.hdim = hdim
        self.nlayers = nlayers
        self.args.bidirectional == True
        self.h_hdim = self.hdim//2 if self.args.bidirectional else self.hdim
        self.embedding = nn.Linear(indim, self.hdim)
        self.rnn = nn.GRU(
            input_size=hdim, 
            hidden_size=self.h_hdim, 
            num_layers=self.nlayers, 
            bidirectional=self.args.bidirectional, 
            batch_first=True)

    def init_hidden(self, bsize):
        (num_directions, h_hdim) = (2, self.hdim//2) if self.args.bidirectional else (1, self.hdim)
        return cuda_if_needed(Variable(torch.zeros(self.nlayers*num_directions, bsize, h_hdim)), self.args)

    def forward(self, x):
        raise NotImplementedError

class SequenceValueFn(GenericRNN):
    def __init__(self, indim, hdim, nlayers, args):
        super(SequenceValueFn, self).__init__(indim, hdim, nlayers, args)
        self.value_head = nn.Linear(hdim, 1)

    def forward(self, x):
        b, t, d = x.size()
        rnn_hid = self.init_hidden(b)
        embedded = self.embedding(x.view(t*b, d)).view(b, t, self.hdim)
        rnn_out = embedded
        rnn_out, rnn_hid = self.rnn(rnn_out, rnn_hid)
        value = self.value_head(rnn_out[:, -1])
        return value

class MultilingualArithmeticPolicy(GenericRNN):
    def __init__(self, indim, hdim, nlayers, num_actions, args):
        # assert args.bidirectional == True
        super(MultilingualArithmeticPolicy, self).__init__(indim, hdim, nlayers, args)
        self.nreducers = args.nreducers
        self.ntranslators = args.ntranslators + 1  # identity
        self.num_actions = self.nreducers + self.ntranslators + 1  # +1 for STOP
        assert self.num_actions == num_actions + 1
        # choose which action: reduce, translate, terminate
        self.action_head = nn.Linear(self.hdim, 3)  # reduce, translate, terminate
        self.translator_head = nn.Linear(self.hdim, self.ntranslators)
        self.reducer_head = nn.Linear(self.hdim, self.nreducers)

    def get_indices(self, state):
        b, t, d = state.size()
        indices = torch.zeros((b,t)).byte()

        if state.is_cuda:
            indices = indices.cuda()

        if t > 1:
            indices[:, 1:-1] = 1
        return indices  # (b, t, 1)

    def forward(self, state):
        """
            state: (b, t, vocabsize+langsize+zsize)

            action_dist: (b, 3)
            rnn_out: (b, t, hdim)
            summarized_rnn_out: (b, hdim)
        """
        b, t, d = state.size()
        rnn_hid = self.init_hidden(b)  # (nlayers * num_directions, b, h_hdim)
        embedded = self.embedding(state.view(b*t, d)).view(b, t, self.hdim)  # (b, t, hdim)
        rnn_out, rnn_hid = self.rnn(embedded, rnn_hid)  # rnn_out: (b, t, hdim), rnn_hid: (nlayers * num_directions, b, h_hdim)
        rnn_out = rnn_out.contiguous()

        forward_rnn_hid = rnn_out[:, -1, :self.h_hdim]
        backward_rnn_hid = rnn_out[:, 0, self.h_hdim:]
        summarized_rnn_out = torch.cat((forward_rnn_hid, backward_rnn_hid), dim=-1)  # (b, hdim)

        # choose to translate, reduce, or stop
        action_scores = self.action_head(summarized_rnn_out)
        action_dist = F.softmax(action_scores, dim=-1)

        return action_dist, rnn_out, summarized_rnn_out

    def get_reducer_dist(self, rnn_out):
        """
            rnn_out: (b, t, hdim)
        """
        b, t = rnn_out.size(0), rnn_out.size(1)
        action_scores = self.reducer_head(rnn_out.view(b*t, -1)).view(b, t, -1)  # (b, t, nreducers)  logits
        # we will mask out the first and last; equivalent to using get_indices
        action_scores = action_scores[:, 1:-1, :]
        action_scores = action_scores.contiguous()
        action_scores = action_scores.view(b, (t-2)*self.nreducers)
        return action_scores

    def get_translator_dist(self, summarized_rnn_out):
        """
            summarized_rnn_out: (b, hdim)
        """
        return self.translator_head(summarized_rnn_out)


    def select_action(self, state):
        b, t, d = state.size()

        action_dist, rnn_out, summarized_rnn_out= self.forward(state)
        """
        action_dist: (b, 3)
        rnn_out: (b, t, hdim)
        summarized_rnn_out: (b, hdim)
        """
        action = sample_from_categorical_dist(action_dist)  # Variable (b)

        if action.data[0] == 2:  # STOP
            stop_dist = cuda_if_needed(Variable(torch.ones(1)), self.args)  # dummy
            secondary_action = sample_from_categorical_dist(stop_dist)
        elif action.data[0] == 1:  # REDUCE
            indices = self.get_indices(state.data)
            if indices.sum() > 0:
                reduction_scores = self.get_reducer_dist(rnn_out)
                reduction_dist = F.softmax(reduction_scores, dim=-1)
            else:
                reduction_dist = cuda_if_needed(Variable(torch.ones(b, 1), volatile=action.volatile), self.args)
            secondary_action = sample_from_categorical_dist(reduction_dist)
        elif action.data[0] == 0:  # TRANSLATE
            translator_scores = self.get_translator_dist(summarized_rnn_out)
            translator_dist = F.softmax(translator_scores, dim=-1)
            secondary_action = sample_from_categorical_dist(translator_dist)
        else:
            assert False
        dist_type = action.data[0]
        if action.data[0] == 2:
            choice_dist = stop_dist.data.cpu().squeeze().numpy()
        elif action.data[0] == 1:
            choice_dist = reduction_dist.data.cpu().squeeze().numpy()
        elif action.data[0] == 0:
            choice_dist = translator_dist.data.cpu().squeeze().numpy()
        else:
            assert False
        meta_dist = action_dist.data.cpu().squeeze().numpy()

        return action.data, secondary_action.data, (dist_type, choice_dist, meta_dist)


    def get_log_prob(self, state, action, secondary_action):
        b, t, d = state.size()
        action_dist, rnn_out, summarized_rnn_out= self.forward(state)
        action_log_prob = logprob_categorical_dist(action_dist, action)
        if action.data[0] == 2:  # STOP
            stop_dist = cuda_if_needed(Variable(torch.ones(1)), self.args)
            secondary_log_prob = logprob_categorical_dist(stop_dist, secondary_action)
        elif action.data[0] == 1:  # REDUCTION
            indices = self.get_indices(state.data)
            if indices.sum() > 0:
                reduction_scores = self.get_reducer_dist(rnn_out)
                reduction_dist = F.softmax(reduction_scores, dim=-1)
            else:
                reduction_dist = cuda_if_needed(Variable(torch.ones(b, 1), volatile=action.volatile), self.args)   
            secondary_log_prob = logprob_categorical_dist(reduction_dist, secondary_action)
        elif action.data[0] == 0:  # TRANSLATE
            translator_scores = self.get_translator_dist(summarized_rnn_out)
            translator_dist = F.softmax(translator_scores, dim=-1)
            secondary_log_prob = logprob_categorical_dist(translator_dist, secondary_action)
        else:
            assert False
        return action_log_prob, secondary_log_prob
