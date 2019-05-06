import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

import cmdp_trainer as ct
from dataloader.multilingual_dataset import ArithmeticLanguageWordEncoding
from learners.multilingual import CRL_MultitaskSequenceAgent
from log import Logger, mkdirp
from networks.encoders import Identity
from sample_episode.cmdp_sample_episode import sample_episode, sample_data

parser = argparse.ArgumentParser(description='Learned Functions')
################################################################################
# Update Every
parser.add_argument('--computation_update', type=int, default=256,
                    help='number of episodes before updating computation (default: 256)')
parser.add_argument('--computation_update_offset', type=int, default=0,
                    help='offset for updating computation (default: 0)')
parser.add_argument('--policy_update', type=int, default=1024,
                    help='number of episodes before updating policy (default: 1024)')
parser.add_argument('--policy_update_offset', type=int, default=0,
                    help='offset for updating policy (default: 0)')

# LR
parser.add_argument('--step_penalty', type=float, default=1e-2,
                    help='step_penalty (default: 1e-2)')
parser.add_argument('--plr', type=float, default=5e-4,
                    help='learning rate (default: 5e-4)')
parser.add_argument('--clr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr-mult-min', type=float, default=1e-4,
                    help='minimum learning rate multiplier')
parser.add_argument('--anneal-policy-lr', action='store_true',
                    help='linearly anneal learning rate for policy')
parser.add_argument('--anneal-comp-lr', action='store_true',
                    help='linearly anneal learning rate for computation')

################################################################################
# Data
parser.add_argument('--bsize', type=int, default=1,
                    help='input batch size for training (default: 1)')
parser.add_argument('--env', type=str, default='arithlang',
                    help='arithlang')
parser.add_argument('--maxterms', nargs='+', type=int, default=[5,5,10],
                    help='number of arithmetic terms: train | val/test | extrap_val/extrap_test')
parser.add_argument('--numrange', nargs='+', type=int, default=[0,10],
                    help='')
parser.add_argument('--ops', type=str, default='+*-',
                    help='types of operations for math')
parser.add_argument('--samplefrom', type=int, default=1e4,
                    help='max number of problems to sample from (1e4)')
parser.add_argument('--episodecap', type=int, default=1e3,
                    help='max number of problems to record (1e3)')
parser.add_argument('--nlang', type=int, default=5,
                    help='number of languages (5)')
parser.add_argument('--pretrainmode', type=str, default='ed',
                    help='ed | ring')

################################################################################
# Model
parser.add_argument('--indim', type=int, default=784,
                    help='hidden dimension of model (default: 784)')
parser.add_argument('--hdimp', type=int, default=128,
                    help='hidden dimension of policy (default: 128)')
parser.add_argument('--hdimf', type=int, default=128,
                    help='hidden dimension of function (default: 128)')
parser.add_argument('--outdim', type=int, default=10,
                    help='hidden dimension of model (default: 10)')

parser.add_argument('--nactions', type=int, default=12,
                    help='number of functions (default: 12)')
parser.add_argument('--nreducers', type=int, default=3,
                    help='number of functions (default: 4)')
parser.add_argument('--ntranslators', type=int, default=8,
                help='number of functions (default: 8)')

parser.add_argument('--nsteps', type=int, default=4,
                    help='number of steps of computation (default: 4)')
parser.add_argument('--model', type=str, default='crl_seq',
                    help='crl_seq')
parser.add_argument('--policylayers', type=int, default=1,
                    help='number of layers of the policy')
parser.add_argument('--functionlayers', type=int, default=1,
                    help='number of layers of the function')
################################################################################
# Algorithm
parser.add_argument('--ppo_optim_epochs', type=int, default=5,
                    help='hidden dimension of model (default: 5)')
parser.add_argument('--ppo_value_iters', type=int, default=1,
                    help='hidden dimension of model (default: 1)')
parser.add_argument('--ppo_minibatch_size', type=int, default=256,
                    help='hidden dimension of model (default: 256)')
parser.add_argument('--ppo_anneal_epochs', action='store_true',
                    help='anneal number of ppo epochs from ppo_optim_epochs to 1')
parser.add_argument('--ppo_clip', type=float, default=0.1,
                    help='hidden dimension of model (default: 0.1)')
parser.add_argument('--entropy_coeff', type=float, default=1e-4,
                    help='hidden dimension of model (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')

################################################################################
# Experimental Config
parser.add_argument('--max_episodes', type=int, default=1e7,
                    help='Maximum number of training episodes')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=1e3, metavar='N',
                    help='interval between training status logs (default: 1000)')
parser.add_argument('--plot', action='store_true', default=True,
                    help='plot')
parser.add_argument('--plot_every', type=int, default=10000,
                    help='plot every (default: 1e5=4)')
parser.add_argument('--track_selected', action='store_true',
                    help='track the function selection')
parser.add_argument('--printf', action='store_true',
                    help='print to file')
parser.add_argument('--save_every', type=int, default=10000,
                    help='number of episodes before saving')
parser.add_argument('--val_every', type=int, default=10000,
                    help='number of episodes before validation')
parser.add_argument('--numval', type=int, default=100,
                    help='number of validation examples')
parser.add_argument('--curr_every', type=int, default=1e5,
                    help='number of episodes before increasing the dataset')
parser.add_argument('--outputdir', type=str, default='runs/arith_verify/crl',
                    help='outputdir')
parser.add_argument('--eval', action='store_true',
                    help='eval mode')
parser.add_argument('--ckpt_every', action='store_true',
                    help='save ckpt every 1e5 iterations')
parser.add_argument('--resume', type=str, default='', metavar='R',
                    help='path of saved model')
parser.add_argument('--debug', action='store_true',
                    help='debug')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

seeder = torch.cuda.manual_seed if args.cuda else torch.manual_seed
np.random.seed(args.seed)
seeder(args.seed)

if args.outputdir != '':
    if not os.path.exists(args.outputdir):
        os.mkdir(args.outputdir)

def process_args(args):
    if args.debug:
        args.max_episodes = 40
        args.val_every = 20
        args.save_every = 20
        args.numval = 10
        args.log_interval = 5
        args.computation_update = 10
        args.policy_update = 10
        args.curr_every = 4
        args.outputdir = 'runs/arith_verify/crl/debug'
    return args

def build_expname(args):
    expname = 'env-{}'.format(args.env)
    expname += '_agent-{}'.format(args.model)
    if args.debug: expname+= '_debug'
    return expname

def main():
    envbuilder = ArithmeticLanguageWordEncoding
    root = 'data'
    env = envbuilder(
        max_terms=args.maxterms, 
        num_range=args.numrange, 
        ops=args.ops, 
        samplefrom=args.samplefrom,
        episodecap=args.episodecap,
        root=root, 
        curr=True,
        nlang=args.nlang
        )
    args.indim = env.vocabsize
    args.outdim = env.vocabsize

    encoder = Identity
    decoder = lambda: Identity(
        predictor=lambda x: torch.max(x.data,-1)[1])
    num_actions = args.nactions  

    expname = build_expname(args)

    main_method = ct.eval if args.eval else ct.train
    args.memory = False
    logdir = expname
    logger = Logger(expname=logdir, logdir=os.path.join(args.outputdir, logdir), params=args)
    agent = CRL_MultitaskSequenceAgent(
        indim=args.indim,
        langsize=env.langsize,
        zsize=env.zsize,           
        hdimp=args.hdimp,           
        hdimf=args.hdimf,           
        outdim=args.indim,          
        num_steps=args.nsteps,      
        num_actions=num_actions,    
        layersp=args.policylayers,  
        layersf=args.functionlayers,
        encoder=encoder, 
        decoder=decoder, 
        args=args,
        relax=True)
    main_method(sample_data, lambda a, e, i, t, m: sample_episode(a, e, i, t, 'crl_seq', m), agent, logger, env, args)

if __name__ == "__main__":
    args = process_args(args)
    main()

