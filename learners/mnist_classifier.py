import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from networks.controllers import CNNPolicy, CNNValueFn
from networks.encoders import Identity
from networks.functions import AffineSTNTransformation, TranslateSTN, ScaleSTN, ConstrainedRotateSTN

from centralized import MarkovAgent

from rb import Memory
import utils as u
import pprint

class CRL_ImageTransforms(MarkovAgent):
    """
        indim: (H, W)
        hdim: (H, W)
        outdim: scalar

        num_steps is not necessary
        num_action is not necessary
    """
    def __init__(self, indim, hdim, outdim, num_steps, num_actions, encoder, decoder, args):
        super(CRL_ImageTransforms, self).__init__(indim, hdim, outdim, num_steps, num_actions, encoder, decoder, args)
        assert indim == hdim

    def initialize_networks(self, indim, hdim, outdim, num_actions, encoder, decoder):
        self.encoder = encoder()  # Identity
        self.actions = nn.ModuleList(
            [ConstrainedRotateSTN(hdim, hdim),
             ScaleSTN(hdim, hdim)])
        for i in range(12):
            self.actions.append(TranslateSTN(hdim, hdim))
        self.actions.append(Identity())

        assert num_actions == len(self.actions)
        self.policy = CNNPolicy(hdim, len(self.actions))
        self.valuefn = CNNValueFn(hdim)
        self.decoder = decoder(outdim)

        computation = [self.encoder]
        computation.append(self.actions)
        if not self.args.pretrain_decoder:
            computation.append(self.decoder)
        self.computation = nn.ModuleList(computation)
        self.assign_model()

        self.learn_computation = True
        self.learn_policy = True

    def assign_model(self):
        self.model = {
            'encoder': self.encoder,
            'policy': self.policy,
            'valuefn': self.valuefn,
            'actions': self.actions,
            'decoder': self.decoder,
            'computation': self.computation
        }
        self.has_computation = len(list(self.computation.parameters())) > 0

    def forward(self, env, state, selected, episode_data):
        assert self.num_actions == len(self.actions)
        num_steps = env.get_composition_depth()
        assert state.size(0) == 1
        for i in range(num_steps):
            action = self.policy.select_action(Variable(state.data, volatile=True))  # Tensor (b)
            log_prob = self.policy.get_log_prob(Variable(state.data), Variable(action))  # Variable (b)
            value = self.valuefn(Variable(state.data))  # Variable (b, 1)
            transformation = self.actions[action[0]]
            next_state = transformation(state)  # Variable (b, hdim)
            done = i == num_steps-1
            mask = 0 if done else 1
            episode_data.append(
                {'state': state.data, 
                 'action': action, 
                 'log_prob': log_prob, 
                 'mask': mask, 
                 'value': value})            
            selected.append((action[0], state.data, transformation.get_parameter()))  # a_t, s_t
            state = next_state  # Variable (b, hdim)
        selected = self.process_selected(selected, state.data)
        return state, selected

    def process_selected(self, selected, last_state):
        selected_actions, selected_states, selected_parameters = zip(*selected)
        selected_states = list(selected_states) + [last_state]
        return selected_states, selected_actions, selected_parameters