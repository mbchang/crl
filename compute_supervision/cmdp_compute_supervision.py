import torch
import torch.nn.functional as F

def compute_loss(agent, out, target):
    if 'mnist_transform' in agent.args.env:
        loss = compute_loss_mnist(out, target)
    elif agent.args.env == 'arithlang':
        loss = compute_loss_arith(out, target)
    else:
        assert False
    return loss

def compute_loss_mnist(out, target):
    b = out.size(0)
    loss = F.nll_loss(out, target.view(b))
    return loss

def compute_loss_arith(out, target):
    b, t, d = out.size()
    out = out.contiguous()
    out = out.view(b*t, d)
    target = target.view(b*t)
    loss = F.cross_entropy(out, target)
    return loss

def compute_reward(agent, out, target, episode_data):
    if 'mnist_transform' in agent.args.env:
        ep_data, correct = compute_reward_mnist(out, target, episode_data)
    elif agent.args.env == 'arithlang':
        ep_data, correct = compute_reward_lang(out, target, episode_data, agent.args.step_penalty)
    else:
        assert False
    return ep_data, correct

def compute_reward_mnist(pred, target, episode_data):
    assert pred.size() == target.size()
    correct = 1 if torch.equal(pred, target.data) else 0
    for i, e in enumerate(episode_data):
        e['reward'] = correct if i == len(episode_data) - 1 else 0
    return episode_data, correct

def compute_reward_lang(pred, target, episode_data, step_penalty):
    target = target.data
    b = pred.size(0)
    diff = pred.view_as(target) - target
    num_incorrect = len(diff.sum(1).nonzero())
    num_correct = b - num_incorrect
    accuracy = float(num_correct) / b  # this is the accuracy that we will use as the reward
    correct = diff[-1][0] == 0
    for i, e in enumerate(episode_data):
        e['reward'] = accuracy if i == len(episode_data) - 1 else -step_penalty
    return episode_data, correct