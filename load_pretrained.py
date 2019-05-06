import torch
import torch.nn as nn
import os

from networks.encoders import Identity
from log import visualize_parameters

def load_checkpoint(ckpt_dir):
    best_loss_candidates = filter(lambda x: 'bestval_loss' in x, os.listdir(ckpt_dir))
    assert len(best_loss_candidates) == 1
    best_loss_model_ckpt = torch.load(os.path.join(ckpt_dir, best_loss_candidates[0]))
    return best_loss_model_ckpt

def get_checkpoint_model(ckpt):
    return ckpt['model']

def load_weights(to_model, from_model, cuda):
    to_model.load_state_dict(from_model)
    if cuda: to_model.cuda()

def load_all_weights(to_models, from_models, cuda):
    for t, ckpt in zip(to_models, from_models):
        load_weights(t, get_checkpoint_model(load_checkpoint(ckpt)), cuda)

def handle_resume(agent, args):

    # if args.resume
    # then try to load the network into agent
    # it won't work if the models don't match.

    if args.resume:
        ckpt_file = args.resume
        ckpt = torch.load(ckpt_file)
        ckpt_model = ckpt['model']
        assert set(agent.model.keys()) == set(ckpt_model.keys())
        for key in agent.model.keys():
            load_weights(agent.model[key], ckpt_model[key], False)
        if args.transfer:
            computation =[agent.encoder]
            if (not args.freeze_functions and not args.hardcode_functions):
                computation.append(agent.actions)
            if not args.pretrain_decoder:
                computation.append(agent.decoder)

            # overwrite
            agent.computation = nn.ModuleList(computation)
            agent.assign_model()
            agent.initialize_optimizers(args)
            agent.initialize_optimizer_schedulers(args)
            if args.cuda:
                print 'CUDA'
                agent.cuda()
                agent.decoder.cuda()
                agent.actions.cuda()

    else:
        if args.model == 'crl_seq':
            pass
        elif args.model == 'crl_image':
            if args.pretrain_decoder:
                print 'PRETRAIN DECODER'
                ckpt = torch.load('pretrained_models/pretrained_mnist_classifier.pth.tar')
                agent.decoder.load_state_dict(ckpt['model'])
                if args.cuda:
                    print 'CUDA'
                    agent.decoder.cuda()
                else:
                    print 'NO CUDA'
        else:
            print 'NOT PRE LOADING'
