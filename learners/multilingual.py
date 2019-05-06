import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from networks.controllers import SequenceValueFn, MultilingualArithmeticPolicy
from networks.encoders import Identity
from networks.functions import OperatorFixedLength, EncoderDecoderRNN, TransformFixedLength, PlainTranslator

from rb import Memory
import utils as u

class CRL_MultitaskSequenceAgent(nn.Module):
    def __init__(self, indim, langsize, zsize, hdimp, hdimf, outdim, num_steps, num_actions, layersp, layersf, encoder, decoder, args, relax):
        super(CRL_MultitaskSequenceAgent, self).__init__()
        self.indim = indim
        self.zsize = zsize
        self.langsize = langsize

        self.hdimp = hdimp
        self.hdimf = hdimf
        self.outdim = outdim

        self.num_steps = num_steps
        self.num_actions = num_actions
        self.nreducers = args.nreducers
        self.ntranslators = args.ntranslators

        self.layersp = layersp
        self.layersf = layersf
        self.outlength = 1

        self.args = args
        self.relaxed = relax

        self.initialize_networks(indim, hdimp, hdimf, outdim, num_actions, layersp, layersf, encoder, decoder)
        self.initialize_memory()
        self.initialize_optimizers(args)
        self.initialize_optimizer_schedulers(args)

    def initialize_networks(self, indim, hdimp, hdimf, outdim, num_actions, layersp, layersf, encoder, decoder):
        controller_input_dim = indim+self.langsize+self.zsize
        self.args.bidirectional = True

        self.encoder = encoder()  # Identity
        self.valuefn = SequenceValueFn(controller_input_dim, hdimp, layersp, self.args)
        ######################################################
        # the number of actions should be 3 reducers + k num_translators + 1 identity + 1 terminate = 3 + k + 1 + 1
        self.policy = MultilingualArithmeticPolicy(controller_input_dim, hdimp, layersp, self.num_actions, self.args)
        ######################################################
        self.translators = nn.ModuleList([PlainTranslator(indim, self.args) for i in xrange(self.ntranslators)] + [Identity()])
        self.reducers = nn.ModuleList([TransformFixedLength(indim, hdimf, outdim, layersf, self.args) for i in range(self.nreducers)])
        self.actions = nn.ModuleList([self.reducers, self.translators])
        ######################################################
        self.decoder = decoder()
        self.computation = nn.ModuleList([self.encoder, self.actions, self.decoder])
        self.model = {'encoder': self.encoder, 
                      'policy': self.policy, 
                      'valuefn': self.valuefn, 
                      'actions': self.actions, 
                      'decoder': self.decoder, 
                      'computation': self.computation}
        self.learn_computation = True
        self.learn_policy = True

    def initialize_memory(self):
        self.replay_buffer = Memory(element='simpletransition')
        self.computation_buffer = Memory(element='inputoutput')

    def initialize_optimizers(self, args):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.plr)
        self.value_optimizer = optim.Adam(self.valuefn.parameters(), lr=args.plr)
        self.computation_optimizer = optim.Adam(self.computation.parameters(), lr=args.clr)
        self.optimizer = {
            'policy_opt': self.policy_optimizer, 
            'value_opt': self.value_optimizer, 
            'computation_opt': self.computation_optimizer}

    def initialize_optimizer_schedulers(self, args):
        if self.args.anneal_policy_lr:
            lr_lambda_policy = lambda epoch: max(1.0 - (float(epoch)/(args.max_episodes / args.policy_update)), args.lr_mult_min)
        else:
            lr_lambda_policy = lambda epoch: 1
        self.po_scheduler = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda_policy)
        self.vo_scheduler = optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda_policy)
        if self.args.anneal_comp_lr:
            lr_lambda_comp = lambda epoch: max(1.0 - (float(epoch)/(args.max_episodes / args.computation_update)), args.lr_mult_min)
        else:
            lr_lambda_comp = lambda epoch: 1
        self.co_scheduler = optim.lr_scheduler.LambdaLR(self.computation_optimizer, lr_lambda_comp)

    def cuda(self):
        self.policy.cuda()
        self.valuefn.cuda()
        self.computation.cuda()

    def encode_policy_in(self, state, target_token, z):
        """
            state: (b, t, vocabsize)
            target_token: (b, langsize)
            z: (b, zsize)

            policy_in: (b, t, vocabsize+langsize+zsize)
        """
        b, t, v = state.size()
        assert target_token.dim() == 2 and z.dim() == 2
        target_token = target_token.unsqueeze(1).repeat(1, t, 1)
        z = z.unsqueeze(1).repeat(1, t, 1)
        policy_in = torch.cat((state, target_token, z), dim=-1)
        return policy_in

    def unpack_state(self, state):
        state, target_token, z = state
        if not isinstance(state, Variable):
            state = Variable(state)
        target_token = Variable(target_token)
        z = Variable(z)
        policy_in_encoder = lambda s: self.encode_policy_in(s, target_token, z)
        return state, policy_in_encoder

    def get_substate_boundaries(self, indices, opidx):
        assert indices.sum() == indices.numel() - 2  # indices should be all ones except 0s at the end
        selected_idx = torch.squeeze(indices).nonzero().squeeze()[opidx]
        indices_list = list(torch.squeeze(indices).cpu().numpy())

        # find boundaries. You should be guaranteed that there are terms
        # begin is the index right on where the term right before opidx begins
        # end is the index right after where the term right after opidx ends

        # there is only one term before this op
        if sum(indices_list[:selected_idx]) == 0:  
            begin = 0
        # assumes no digit
        else:
            begin = selected_idx - 1

        # there is only one term after this op
        if sum(indices_list[selected_idx+1:]) == 0:
            end = len(indices_list)  
        else:
            end = selected_idx + 2
        return begin, selected_idx, end

    def isolate_index(self, state, substate_boundaries):
        """
            state: (b, t, d)
            indicies: (b, d)
            opidx: 1
        """
        assert state.size(0) == 1
        begin, selected_idx, end = substate_boundaries
        substate = state[:, begin:end, :]  # Variable FloatTensor (b, subexp_length, indim)
        return substate

    def update_state(self, state, substate_boundaries, substate_transformation, args):
        assert isinstance(state, torch.autograd.variable.Variable)
        assert isinstance(substate_transformation, torch.autograd.variable.Variable)
        begin, selected_idx, end = substate_boundaries

        # NOTE: the substate_transformation is not one hot!!!
        b = substate_transformation.size(0)
        d = state.size(-1)

        substate_transformation = substate_transformation.view(b, -1, d)
        if begin == 0 and end == state.size(1):
            transformed_state = substate_transformation
        elif begin == 0:
            transformed_state = torch.cat((substate_transformation, state[:,end:]), dim=1)
        elif end == state.size(1):
            transformed_state = torch.cat((state[:,:begin], substate_transformation), dim=1)
        else:
            transformed_state = torch.cat((state[:,:begin], substate_transformation, state[:,end:]), dim=1)

        return transformed_state

    def run_policy(self, state):
        action, secondary_action, choice_dist_info = self.policy.select_action(Variable(state.data, volatile=True))
        action_logprob, secondary_log_prob = self.policy.get_log_prob(Variable(state.data), Variable(action), Variable(secondary_action))
        value = self.valuefn(Variable(state.data))  # (b, 1)
        return action, secondary_action, action_logprob, secondary_log_prob, value, choice_dist_info

    def run_functions(self, state, indices, a, opidx, env):
        substate_boundaries = self.get_substate_boundaries(indices, opidx)
        substate = self.isolate_index(state, substate_boundaries)
        substate_transformation_logits, substate_transformation = self.actions[a](substate)  # Variable (b, indim)
        return substate_transformation_logits, substate_transformation, substate_boundaries

    def get_meta_selected(self, selected):
        return [s[0] for s in selected]

    def get_sub_selected(self, selected):
        return [self.get_reducer_and_idx(v[1], i) if v[0] == 1 else v[1] for i, v in enumerate(selected)]

    def forward(self, env, state, selected, episode_data):
        state, policy_in_encoder = self.unpack_state(state)
        env.add_exp_str(env.get_exp_str(torch.squeeze(state.data.clone(), dim=0)))
        while True: # Can potentially infinite loop. Hopefully the agent realizes it should terminate.
            policy_in = policy_in_encoder(state)
            action, secondary_action, action_logprob, secondary_log_prob, value, choice_dist_info = self.run_policy(policy_in)
            a, sa = action[0], secondary_action[0]
            if a == 2:  # STOP
                if state.size(1) > 1:
                    done = False
                    next_state = state
                else:
                    done = True
                    env.add_exp_str('END')
            else:
                if a == 1:  # REDUCE
                    r, idx = self.get_reducer_and_idx(sa)
                    indices = self.policy.get_indices(state)
                    if indices.sum() == 0:
                        next_state = state
                    else:
                        next_state, substate_transformation_logits = self.apply_reducer(r, idx, indices, state)
                elif a == 0:  # TRANSLATE
                    assert isinstance(self.translators[-1], Identity)
                    if sa == len(self.translators)-1:  # Identity
                        next_state = state # and substate_transformation_logits remain the same
                        """
                        it will never be the case that substate_transformation_logits will not be 
                        defined when we have Identity and we are expected to output because it will 
                        have to keep on going before it will be finally reduced
                        """
                        # 
                    else:
                        next_state, substate_transformation_logits = self.apply_translator(sa, state)
                else:
                    assert False
                done = False
            selected.append((a, sa))

            mask = 0 if done else 1
            episode_data.append(
                {'state': policy_in.data,
                 'action': (action, secondary_action),
                 'log_prob': (action_logprob, secondary_log_prob),
                 'mask': mask,
                 'value': value,
                 'choice_dist_info': choice_dist_info
                })

            if done:
                break
            else:
                state = next_state
                env.add_exp_str(env.get_exp_str(torch.squeeze(state.data.clone(), dim=0)))
        state = substate_transformation_logits
        return substate_transformation_logits, selected

    def improve_actions(self, retain_graph=False):
        batch = self.computation_buffer.sample()
        loss = list(batch.loss)
        loss = loss[0] if len(loss) == 1 else torch.mean(torch.cat(loss))
        if loss.requires_grad:
            self.computation_optimizer.zero_grad()
            loss.backward(retain_graph=retain_graph)
            self.computation_optimizer.step()

    def apply_reducer(self, r, idx, indices, state):
        substate_boundaries = self.get_substate_boundaries(indices, idx)
        substate = self.isolate_index(state, substate_boundaries)
        substate_transformation_logits, substate_transformation = self.reducers[r](substate)
        next_state = self.update_state(state, substate_boundaries, substate_transformation, self.args)
        if substate_transformation_logits.dim() < 3:
            substate_transformation_logits = substate_transformation_logits.unsqueeze(1)
        return next_state, substate_transformation_logits

    def apply_translator(self, translator_idx, state):
        substate_transformation_logits = self.translators[translator_idx](state)  # logits
        transformation = F.softmax(substate_transformation_logits, dim=-1)
        return transformation, substate_transformation_logits

    def get_reducer_and_idx(self, reducer_idx, step=None):
        num_reducers = len(self.reducers)
        idx = reducer_idx // num_reducers  # the row
        r = reducer_idx % num_reducers  # the column
        return r, idx

    def unpack_ppo_batch(self, batch):
        """
                        batch.state: tuple of num_episodes of FloatTensor (1, t, state-dim), where t is variable
            batch.action: tuple of num_episodes of tuples of (LongTensor (1), LongTensor (1)) for (action, index)
            batch.reward: tuple of num_episodes scalars in {0,1}
            batch.masks: tuple of num_episodes scalars in {0,1}
            batch.value: tuple of num_episodes Variable FloatTensor of (1,1)
            batch.logprob: tuple of num_episodes of tuples of (FloatTensor (1), FloatTensor (1)) for (action_logprob, index_logprob)

            states is not a variable
            actions is not a variable: tuple of length (B) of LongTensor (1)
            secondary_actions is not a variable: tuple of length (B) of LongTensor (1)
            action_logprobs is not a variable (B)
            secondary_log_probs is not a variable (B)
            values is not a variable: (B, 1)
            rewards is not a variable: FloatTensor (B)
            masks is not a variable: FloatTensor (B)
            perm_idx is a tuple
            group_idx is an array 
        """
        lengths = [e.size(1) for e in batch.state]
        perm_idx, sorted_lengths = u.sort_decr(lengths)
        group_idx, group_lengths = u.group_by_element(sorted_lengths)

        states = batch.state  # tuple of num_episodes of FloatTensor (1, t, state-dim), where t is variable
        actions, secondary_actions = zip(*batch.action)
        action_logprobs, secondary_log_probs = zip(*batch.logprob)
        action_logprobs = torch.cat(action_logprobs).data  # FloatTensor (B)
        secondary_log_probs = torch.cat(secondary_log_probs).data  # FloatTensor (B)
        values = torch.cat(batch.value).data  # FloatTensor (B, 1)
        rewards = u.cuda_if_needed(torch.from_numpy(np.stack(batch.reward)).float(), self.args)  # FloatTensor (b)
        masks = u.cuda_if_needed(torch.from_numpy(np.stack(batch.mask)).float(), self.args)  # FloatTensor (b)
        return states, actions, secondary_actions, action_logprobs, secondary_log_probs, values, rewards, masks, perm_idx, group_idx

    def estimate_advantages(self, rewards, masks, values, gamma, tau):
        """
            returns: (B, 1)
            deltas: (B, 1)
            advantages: (B, 1)
            mask: (B)
            values: (B, 1)
        """
        tensor_type = type(rewards)
        returns = tensor_type(rewards.size(0), 1)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages, returns

    def improve_policy_ppo(self):
        optim_epochs = self.args.ppo_optim_epochs  # can anneal this
        minibatch_size = self.args.ppo_minibatch_size
        num_value_iters = self.args.ppo_value_iters
        clip_epsilon = self.args.ppo_clip
        gamma = self.args.gamma
        tau = 0.95
        l2_reg = 1e-3

        batch = self.replay_buffer.sample()

        all_states, all_actions, all_indices, all_fixed_action_logprobs, all_fixed_index_logprobs, all_values, all_rewards, all_masks, perm_idx, group_idx = self.unpack_ppo_batch(batch)
        all_advantages, all_returns = self.estimate_advantages(all_rewards, all_masks, all_values, gamma, tau) # (b, 1) (b, 1)

        # permute everything by length
        states_p, actions_p, indices_p, returns_p, advantages_p, fixed_action_logprobs_p, fixed_index_logprobs_p = map(
            lambda x: u.permute(x, perm_idx), [all_states, all_actions, all_indices, all_returns, all_advantages, all_fixed_action_logprobs, all_fixed_index_logprobs])

        # group everything by length
        states_g, actions_g, indices_g, returns_g, advantages_g, fixed_action_logprobs_g, fixed_index_logprobs_g = map(
            lambda x: u.group_by_indices(x, group_idx), [states_p, actions_p, indices_p, returns_p, advantages_p, fixed_action_logprobs_p, fixed_index_logprobs_p])
        
        for j in range(optim_epochs):

            for grp in range(len(group_idx)):
                states = torch.cat(states_g[grp], dim=0)  # FloatTensor (g, grp_length, indim)
                actions = torch.cat(actions_g[grp])  # LongTensor (g)
                indices = torch.cat(indices_g[grp])  # LongTensor (g)
                returns = torch.cat(returns_g[grp])  # FloatTensor (g)
                advantages = torch.cat(advantages_g[grp])  # FloatTensor (g)
                fixed_action_logprobs = u.cuda_if_needed(torch.FloatTensor(fixed_action_logprobs_g[grp]), self.args)  # FloatTensor (g)
                fixed_index_logprobs = u.cuda_if_needed(torch.FloatTensor(fixed_index_logprobs_g[grp]), self.args)  # FloatTensor (g)

                for x in [states, actions, indices, returns, advantages, fixed_action_logprobs, fixed_index_logprobs]:
                    assert not isinstance(x, torch.autograd.variable.Variable)

                perm = np.random.permutation(range(states.shape[0]))
                perm = u.cuda_if_needed(torch.LongTensor(perm), self.args)

                states, actions, indices, returns, advantages, fixed_action_logprobs, fixed_index_logprobs = \
                    states[perm], actions[perm], indices[perm], returns[perm], advantages[perm], fixed_action_logprobs[perm], fixed_index_logprobs[perm]

                optim_iter_num = int(np.ceil(states.shape[0] / float(minibatch_size)))
                for i in range(optim_iter_num):
                    ind = slice(i * minibatch_size, min((i + 1) * minibatch_size, states.shape[0]))
                    
                    states_b, actions_b, indices_b, advantages_b, returns_b, fixed_action_logprobs_b, fixed_index_logprobs_b = \
                        states[ind], actions[ind], indices[ind], advantages[ind], returns[ind], fixed_action_logprobs[ind], fixed_index_logprobs[ind]

                    self.ppo_step(num_value_iters, states_b, actions_b, indices_b, returns_b, advantages_b, fixed_action_logprobs_b, fixed_index_logprobs_b,
                        1, self.args.plr, clip_epsilon, l2_reg)

    def ppo_step(self, num_value_iters, states, actions, indices, returns, advantages, fixed_action_logprobs, fixed_index_logprobs, lr_mult, lr, clip_epsilon, l2_reg):
        clip_epsilon = clip_epsilon * lr_mult

        """update critic"""
        values_target = Variable(u.cuda_if_needed(returns, self.args))  # (mb, 1)
        for k in range(num_value_iters):
            values_pred = self.valuefn(Variable(states))  # (mb, 1)
            value_loss = (values_pred - values_target).pow(2).mean()
            # weight decay
            for param in self.valuefn.parameters():
                value_loss += param.pow(2).sum() * l2_reg
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        """update policy"""
        advantages_var = Variable(u.cuda_if_needed(advantages, self.args)).view(-1)  # (mb)

        ########################################
        perm_idx, sorted_actions = u.sort_decr(actions)
        inverse_perm_idx = u.invert_permutation(perm_idx)
        group_idx, group_actions = u.group_by_element(sorted_actions)

        # permute everything by action type
        states_ap, actions_ap, indices_ap = map(lambda x: u.permute(x, perm_idx), [states, actions, indices])

        # group everything by action type
        states_ag, actions_ag, indices_ag = map(lambda x: u.group_by_indices(x, group_idx), [states_ap, actions_ap, indices_ap])

        action_logprobs, index_logprobs = [], []
        for grp in xrange(len(group_idx)):
            states_grp = torch.stack(states_ag[grp])  # (g, grp_length, indim)
            actions_grp = torch.LongTensor(np.stack(actions_ag[grp]))  # (g)
            indices_grp = torch.LongTensor(np.stack(indices_ag[grp]))  # (g)

            actions_grp = u.cuda_if_needed(actions_grp, self.args)
            indices_grp = u.cuda_if_needed(indices_grp, self.args)

            alp, ilp = self.policy.get_log_prob(Variable(states_grp), Variable(actions_grp), Variable(indices_grp))

            action_logprobs.append(alp)
            index_logprobs.append(ilp)

        action_logprobs = torch.cat(action_logprobs)
        index_logprobs = torch.cat(index_logprobs)

        # unpermute
        inverse_perm_idx = u.cuda_if_needed(torch.LongTensor(inverse_perm_idx), self.args)
        action_logprobs = action_logprobs[inverse_perm_idx]
        index_logprobs = index_logprobs[inverse_perm_idx]
        ########################################
        ratio = torch.exp(action_logprobs + index_logprobs - Variable(fixed_action_logprobs) - Variable(fixed_index_logprobs))
        surr1 = ratio * advantages_var  # (mb)
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var  # (mb)
        policy_surr = -torch.min(surr1, surr2).mean()
        self.policy_optimizer.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 40)
        self.policy_optimizer.step()
