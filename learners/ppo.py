import numpy as np
import torch
from torch.autograd import Variable
import utils as u

class PPO():
    def __init__(self, policy, policy_optimizer, valuefn, value_optimizer, replay_buffer, gamma, optim_epochs=5, minibatch_size=64, value_iters=1, clip_epsilon=0.1, lr_mult=1, tau=0.95, l2_reg=1e-3, entropy_coeff=1e-4):
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.valuefn = valuefn
        self.value_optimizer = value_optimizer
        self.replay_buffer = replay_buffer

        self.optim_epochs = optim_epochs
        self.minibatch_size = minibatch_size
        self.value_iters = value_iters
        self.clip_epsilon = clip_epsilon
        self.tau = tau
        self.l2_reg = l2_reg
        self.entropy_coeff = entropy_coeff

        self.gamma = gamma
        self.lr_mult = lr_mult

    def unpack_ppo_batch(self, batch):
        """
            states: FloatTensor (b, hdim)
            actions: LongTensor (b)
            rewards: FloatTensor (b)
            masks: FloatTensor (b)
        """
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.from_numpy(np.stack(batch.reward)).float()
        masks = torch.from_numpy(np.stack(batch.mask)).float()
        return states, actions, rewards, masks

    def estimate_advantages(self, rewards, masks, values):
        """ GAE
            rewards: (B,)
            masks: (B,)
            values: (B, 1)

            returns: (B, 1)
            deltas: (B, 1)
            advantages: (B, 1)
        """
        tensor_type = type(rewards)
        returns = tensor_type(rewards.size(0), 1)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages, returns

    def improve(self, args):

        ###########################################################################

        batch = self.replay_buffer.sample()

        states, actions, rewards, masks = self.unpack_ppo_batch(batch)  # none of these are Variables, so we are good        
        states, actions, rewards, masks = map(lambda x: u.cuda_if_needed(x, args), (states, actions, rewards, masks))

        values = self.valuefn(Variable(states, volatile=True)).data  # (b, 1)
        fixed_log_probs = self.policy.get_log_prob(Variable(states, volatile=True), Variable(actions)).data  # (b)

        advantages, returns = self.estimate_advantages(rewards, masks, values) # (b, 1) (b, 1)

        optim_iter_num = int(np.ceil(states.shape[0] / float(self.minibatch_size)))
        for j in range(self.optim_epochs):
            perm = np.random.permutation(range(states.shape[0]))
            perm = u.cuda_if_needed(torch.LongTensor(perm), args)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm], actions[perm], returns[perm], advantages[perm], fixed_log_probs[perm]

            for i in range(optim_iter_num):
                ind = slice(i * self.minibatch_size, min((i + 1) * self.minibatch_size, states.shape[0]))
                
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                minibatch = {
                    'states': states_b,
                    'actions': actions_b,
                    'returns': returns_b,
                    'advantages': advantages_b,
                    'fixed_log_probs': fixed_log_probs_b
                }

                self.ppo_step(minibatch=minibatch, args=args)

    def ppo_step(self, minibatch, args):

        states = minibatch['states']
        actions = minibatch['actions']
        returns = minibatch['returns']
        advantages = minibatch['advantages']
        fixed_log_probs = minibatch['fixed_log_probs']

        ###########################################################################

        self.clip_epsilon = self.clip_epsilon * self.lr_mult  # NOTE: this is deprecated. Set self.lr_mult=1. We can anneal based on pytorch's scheduler.

        """update critic"""
        values_target = Variable(u.cuda_if_needed(returns, args))  # (mb, 1)
        for k in range(self.value_iters):
            values_pred = self.valuefn(Variable(states))  # (mb, 1)
            value_loss = (values_pred - values_target).pow(2).mean()
            # weight decay
            for param in self.valuefn.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        """update policy"""
        advantages_var = Variable(u.cuda_if_needed(advantages, args)).view(-1)  # (mb)
        log_probs = self.policy.get_log_prob(Variable(states), Variable(actions))  # (mb)
        probs = torch.exp(log_probs)  # (mb)
        entropy = torch.sum(-(log_probs * probs))  # (1)
        ratio = torch.exp(log_probs - Variable(fixed_log_probs))  # (mb)
        surr1 = ratio * advantages_var  # (mb)
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_var  # (mb)
        policy_surr = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy  # (1)  subtract entropy!
        self.policy_optimizer.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 40)
        self.policy_optimizer.step()
