import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pprint

from rb import Memory
import utils as u
from ppo import PPO


class BaseAgent(nn.Module):
    def __init__(self, indim, hdim, outdim, num_steps, num_actions, encoder, decoder, args):
        super(BaseAgent, self).__init__()
        self.indim = indim
        self.hdim = hdim
        self.outdim = outdim
        self.num_steps = num_steps
        self.num_actions = num_actions
        self.args = args

        self.initialize_networks(indim, hdim, outdim, num_actions, encoder, decoder)
        self.initialize_memory()
        self.initialize_optimizers(args)
        self.initialize_optimizer_schedulers(args)
        self.initialize_rl_alg(args)

    def initialize_networks(self, indim, hdim, outdim, num_actions, encoder, decoder):
        raise NotImplementedError

    def initialize_memory(self):
        self.replay_buffer = Memory(element='simpletransition')
        self.computation_buffer = Memory(element='inputoutput')

    def initialize_optimizers(self, args):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.plr)
        self.value_optimizer = optim.Adam(self.valuefn.parameters(), lr=args.plr)
        if self.has_computation:
            self.computation_optimizer = optim.Adam(self.computation.parameters(), lr=args.clr)
            self.optimizer = {'policy_opt': self.policy_optimizer, 'value_opt': self.value_optimizer, 'computation_opt': self.computation_optimizer}
        else:
            self.optimizer = {'policy_opt': self.policy_optimizer, 'value_opt': self.value_optimizer}

    def initialize_rl_alg(self, args):
        hyperparams = {
            'optim_epochs': self.args.ppo_optim_epochs,
            'minibatch_size': self.args.ppo_minibatch_size,
            'gamma': self.args.gamma,
            'value_iters': self.args.ppo_value_iters,
            'clip_epsilon': self.args.ppo_clip,
            'entropy_coeff': self.args.entropy_coeff,
        }

        self.rl_alg = PPO(
            policy=self.policy, 
            policy_optimizer=self.policy_optimizer, 
            valuefn=self.valuefn, 
            value_optimizer=self.value_optimizer, 
            replay_buffer=self.replay_buffer,
            **hyperparams)

    def initialize_optimizer_schedulers(self, args):
        if not self.args.anneal_policy_lr: assert self.args.anneal_policy_lr_gamma == 1
        self.po_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=args.anneal_policy_lr_step, gamma=args.anneal_policy_lr_gamma, last_epoch=-1)
        self.vo_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=args.anneal_policy_lr_step, gamma=args.anneal_policy_lr_gamma, last_epoch=-1)
        if self.has_computation:
            if not self.args.anneal_comp_lr: assert self.args.anneal_comp_lr_gamma == 1 
            self.co_scheduler = optim.lr_scheduler.StepLR(self.computation_optimizer, step_size=args.anneal_comp_lr_step, gamma=args.anneal_comp_lr_gamma, last_epoch=-1)

    def cuda(self):
        self.policy.cuda()
        self.valuefn.cuda()
        self.computation.cuda()

    def forward(self, x):
        raise NotImplementedError

    def compute_returns(self, rewards):
        returns = []
        prev_return = 0
        for r in rewards[::-1]:
            prev_return = r + self.args.gamma * prev_return
            returns.insert(0, prev_return)
        return returns

    def improve_actions(self, retain_graph=False):
        batch = self.computation_buffer.sample()
        loss = list(batch.loss)  # these are all the same
        loss = loss[0] if len(loss) == 1 else torch.mean(torch.cat(loss))  # these are all the same
        if loss.requires_grad:
            self.computation_optimizer.zero_grad()
            loss.backward(retain_graph=retain_graph)
            self.computation_optimizer.step()

    def improve_policy_ac(self, retain_graph=False):
        batch = self.replay_buffer.sample()
        b_lp = batch.logprob  # tuple length num_steps of Variable (b)
        b_rew = list(batch.reward)  # tuple length num_steps
        b_v = batch.value  # tuple length num_steps of Variable (b)
        b_ret = self.compute_returns(b_rew)
        ac_step(b_lp, b_v, b_ret, self.policy_optimizer, self.value_optimizer, self.args, retain_graph)

    def improve_policy_ppo(self):
        self.rl_alg.improve(args=self.args)

class MarkovAgent(BaseAgent):
    def __init__(self, indim, hdim, outdim, num_steps, num_actions, encoder, decoder, args):
        super(MarkovAgent, self).__init__(indim, hdim, outdim, num_steps, num_actions, encoder, decoder, args)

    def forward(self, env, state, selected, episode_data):
        for i in range(self.num_steps):
            ###############################################################
            action = self.policy.select_action(Variable(state.data, volatile=True))  # Tensor (b)
            log_prob = self.policy.get_log_prob(Variable(state.data), Variable(action))  # Variable (b)
            value = self.valuefn(Variable(state.data))  # Variable (b, 1)
            ###############################################################
            next_state = self.actions[action[0]](state)  # Variable (b, hdim)  # HACK because we assume action is just a scalar
            ###############################################################
            done = i == self.num_steps-1
            mask = 0 if done else 1
            episode_data.append({'state': state.data, 'action': action, 'log_prob': log_prob, 'mask': mask, 'value': value})
            selected.append(action[0])
            state = next_state  # Variable (b, hdim)
        return state