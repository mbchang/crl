import numpy as np
import torch
from torch.autograd import Variable
from compute_supervision.cmdp_compute_supervision import compute_loss, compute_reward
import utils as u
import dataloader.datautils as du

def sample_data(episode_sampler, create_batch, bsize, agent, env, args, mode):
    assert args.bsize == 1
    initial, target = create_batch(env, bsize, mode, args)
    loss, selected, pred, correct, episode_data = episode_sampler(agent, env, initial, target, mode)
    reformatted_pred = env.decode_tokens(pred[-1].cpu().numpy())
    reformatted_target = env.decode_tokens(target.data[-1].cpu().numpy())
    return loss, selected, reformatted_pred, reformatted_target, correct, episode_data

def sample_episode(agent, env, initial, target, method, mode):
    selected = []
    episode_data = []

    # sample episode
    state = agent.encoder(initial)  # Variable (b, hdim)
    state, selected = agent.forward(env, state, selected, episode_data)
    out, pred = agent.decoder(state)  # Variable (b, outdim) (b, 1)

    # compute rewards and then push to replay buffer
    episode_data, correct = compute_reward(agent, pred, target, episode_data)
    if mode == 'train' and agent.learn_policy:
        for i, e in enumerate(episode_data):
            agent.replay_buffer.push(e['state'], e['action'], e['log_prob'], e['mask'], e['reward'], e['value'])

    # compute loss and push to computation buffer
    loss = compute_loss(agent, out, target)
    if mode == 'train' and agent.learn_computation:
        agent.computation_buffer.push(loss)
    loss = float(loss.data.cpu().numpy())
    return loss, selected, pred, correct, episode_data