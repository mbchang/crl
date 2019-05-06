import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import pprint

from env_config import load_image_xforms_env, create_logger
import mnist_trainer as ct
from dataloader.mnist_dataset import MNIST, load_mnist_datasets
from learners.mnist_classifier import CRL_ImageTransforms
from log import Logger
from networks.encoders import Identity, CNN64fc_8
from sample_episode.cmdp_sample_episode import sample_episode, sample_data

import copy

parser = argparse.ArgumentParser(description='CRL MNIST Classification')
################################################################################
# Data
parser.add_argument('--bsize', type=int, default=1,
                    help='input batch size for training (default: 1)')
parser.add_argument('--env', type=str, default='mnist_transform',
                    help='mnist_transform')
parser.add_argument('--cdepth', type=int, default=1,
                    help='composition depth')
parser.add_argument('--splittype', type=str, default='all',
                    help='all | 5c1 | 5c2 | 4c1 | 4c2')
parser.add_argument('--num_transfer', type=float, default=1.,
                    help='percentage of original dataset to train on during transfer')

################################################################################
# Model
parser.add_argument('--indim', type=int, default=784,
                    help='hidden dimension of model (default: 784)')
parser.add_argument('--hdim', type=int, default=128,
                    help='hidden dimension (default: 128)')
parser.add_argument('--outdim', type=int, default=10,
                    help='hidden dimension of model (default: 10)')
parser.add_argument('--nactions', type=int, default=15,
                    help='number of functions (default: 15)')
parser.add_argument('--nsteps', type=int, default=1,
                    help='number of steps of computation (default: 1)')
parser.add_argument('--model', type=str, default='crl_image',
                    help='markov | memory | act | affine')
parser.add_argument('--pretrain_decoder', action='store_true', default=True,
                    help='pretrain_decoder')

################################################################################
# Algorithm
parser.add_argument('--ppo', action='store_true', default=True,
                    help='PPO')
parser.add_argument('--ppo_optim_epochs', type=int, default=5,
                    help='hidden dimension of model (default: 5)')
parser.add_argument('--ppo_value_iters', type=int, default=1,
                    help='hidden dimension of model (default: 1)')
parser.add_argument('--ppo_minibatch_size', type=int, default=64,
                    help='hidden dimension of model (default: 64)')
parser.add_argument('--ppo_anneal_epochs', action='store_true',
                    help='anneal number of ppo epochs from ppo_optim_epochs to 1')
parser.add_argument('--ppo_clip', type=float, default=0.1,
                    help='hidden dimension of model (default: 0.1)')
parser.add_argument('--entropy_coeff', type=float, default=1e-2,
                    help='hidden dimension of model (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')

# Update Every
parser.add_argument('--computation_update', type=int, default=64,
                    help='number of episodes before updating computation (default: 256)')
parser.add_argument('--computation_update_offset', type=int, default=0,
                    help='offset for updating computation (default: 0)')
parser.add_argument('--policy_update', type=int, default=256,
                    help='number of episodes before updating controller (default: 1024)')
parser.add_argument('--policy_update_offset', type=int, default=0,
                    help='offset for updating controller (default: 0)')

# LR
parser.add_argument('--plr', type=float, default=5e-5,
                    help='controller learning rate (default: 5e-4)')
parser.add_argument('--clr', type=float, default=5e-4,
                    help='functions learning rate (default: 1e-3)')

parser.add_argument('--lr-mult-min', type=float, default=1e-4,
                    help='minimum learning rate multiplier')

parser.add_argument('--anneal-policy-lr', action='store_true',
                    help='linearly anneal learning rate for controller')
parser.add_argument('--anneal-policy-lr-after', type=int, default=0,
                    help='How many episodes before beginning to anneal plr')

parser.add_argument('--anneal-policy-lr-step', type=int, default=1e2,
                    help='How many iterations before multiplying lr by anneal-policy-lr-gamma')
parser.add_argument('--anneal-policy-lr-gamma', type=float, default=1,
                    help='exponential decay rate for policy-lr')

parser.add_argument('--anneal-comp-lr', action='store_true',
                    help='linearly anneal learning rate for computation')
parser.add_argument('--anneal-comp-lr-after', type=int, default=0,
                    help='How many episodes before beginning to anneal clr')

parser.add_argument('--anneal-comp-lr-step', type=int, default=1e2,
                    help='How many iterations before multiplying lr by anneal-comp-lr-gamma')
parser.add_argument('--anneal-comp-lr-gamma', type=float, default=1,
                    help='exponential decay rate for comp-lr')

################################################################################
# Experimental Config
parser.add_argument('--max_episodes', type=int, default=1e7,
                    help='Maximum number of training episodes')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--track_selected', action='store_true',
                    help='track the function selection')
parser.add_argument('--visualize_selected', action='store_true', default=True,
                    help='visualize the transformations')

parser.add_argument('--log_interval', type=int, default=1000,
                    help='interval between training status logs (default: 100)')
parser.add_argument('--save_every', type=int, default=10000,
                    help='number of episodes before saving')
parser.add_argument('--val_every', type=int, default=10000,
                    help='number of episodes before validation')
parser.add_argument('--numval', type=int, default=100,
                    help='number of validation examples')
parser.add_argument('--plot_every', type=int, default=10000,
                    help='plot every (default: 1e5=4)')
parser.add_argument('--curr_every', type=int, default=30000,
                    help='curr every (default: 30000)')

parser.add_argument('--outputdir', type=str, default='runs/image_verify/crl',
                    help='outputdir')
parser.add_argument('--printf', action='store_true',
                    help='print to file')

parser.add_argument('--curr', action='store_true', default=True,
                    help='curr mode')
parser.add_argument('--eval', action='store_true',
                    help='eval mode')
parser.add_argument('--transfer', action='store_true',
                    help='transfer mode')
parser.add_argument('--resume', type=str, default='',
                    help='.tar path of saved model')
parser.add_argument('--debug', action='store_true',
                    help='debug')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

seeder = torch.cuda.manual_seed if args.cuda else torch.manual_seed
np.random.seed(args.seed)
seeder(args.seed)

assert args.val_every == args.save_every
args.memory=False

if args.outputdir != '':
    if not os.path.exists(args.outputdir):
        os.mkdir(args.outputdir)

def process_args(args):
    if args.debug:
        args.max_episodes = 36
        args.log_interval = 1
        args.computation_update = 3
        args.policy_update = 3
        args.val_every = 9
        args.save_every = 9
        args.curr_every = 9
        args.anneal_policy_lr = True
        args.anneal_policy_lr_step = 2
        args.anneal_policy_lr_gamma = 0.9
        args.anneal_policy_lr_after = 9
        args.anneal_comp_lr = True
        args.anneal_comp_lr_step = 3
        args.anneal_comp_lr_gamma = 0.1
        args.anneal_comp_lr_after = 6
        args.outputdir = 'debug_verify/image_verify/crl'
    return args

def build_expname(args):
    expname = 'env-{}'.format(args.env)
    expname += '_agent-{}'.format(args.model)
    if args.debug: expname+= '_debug'
    return expname

def create_env(args):
    env = load_image_xforms_env(args, args.cuda, mix_in_normal=False)
    args.debug_models = {}
    args.imgxform_models = {}
    args.drastic_models = {
        'crl_image': CRL_ImageTransforms,
    }  # seems same as imgxforms
    return env

def create_agent(args):
    if args.drastic_models:
        assert not args.debug_models and not args.imgxform_models
        encoder = Identity
        decoder = lambda o: CNN64fc_8(o,
            activation=lambda x: F.log_softmax(x, dim=-1),
            predictor = lambda x: x.data.topk(1)[1])

        # Hacky
        args.indim = (64, 64)
        args.hdim = (64, 64)
        agent = args.drastic_models [args.model](
                indim=args.indim, 
                hdim=args.hdim, 
                outdim=args.outdim, 
                num_steps=args.nsteps, 
                num_actions=args.nactions, 
                encoder=encoder, 
                decoder=decoder, 
                args=args)
    else:
        assert False
    return agent

def main(args):
    args = process_args(args)
    logger = create_logger(build_expname, args)
    env = create_env(args)
    agent = create_agent(args)
    main_method = ct.eval if args.eval else ct.train
    main_method(sample_data, lambda a, e, i, t, m: sample_episode(a, e, i, t, args.model, m), agent, logger, env, args)

if __name__ == "__main__":
    main(args)

