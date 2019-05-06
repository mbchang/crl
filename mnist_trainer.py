import numpy as np
import operator
import os
import shutil
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import dataloader.datautils as du
from dataloader.numerical_dataset import BaseArithmetic
import itertools
from itertools import count
from load_pretrained import handle_resume
from log import RunningAverage, visualize_parameters
from utils import cuda_if_needed, printf

from trainer import initialize_logger, to_cpu, base_eval, base_plot

def create_batch_transforms(env, bsize, mode, args):
    """
        initial: (bsize, nchannels, H, W)
        target: (bsize, 1)

        env.get_xform_combo_info() returns
            {'forward': tuple of lambda transformations),
             'inverse': tuple of lambda transformations),
             'name': which dataset the example came from,
             'ids': tuple of the ids of the Transform objects}
    """
    volatile = mode != 'train'
    initial, target = env.reset(mode, bsize)
    initial, target = Variable(initial, volatile=volatile), Variable(target, volatile=volatile)
    return initial, target

def validate(data_sampler, episode_sampler, agent, env, args, logger, mode, i_episode):
    val_losses = []
    val_moves = []
    val_rewards = []
    val_accuracy = []
    visualize_data = []
    for i in range(args.numval):
        val_loss, val_selected, val_pred, val_target, val_correct, val_episode_data = data_sampler(
            episode_sampler, create_batch_transforms, args.bsize, agent, env, args, mode=mode)
        val_losses.append(val_loss)
        val_moves.append(len(val_episode_data))
        val_rewards.append(sum([e['reward'] for e in val_episode_data]))
        val_accuracy.append(val_correct)
        val_trace = env.get_trace()

        if args.visualize_selected:
            if i_episode % (10*args.val_every) == 0:
                if i < 10:
                    logger.visualize_transformations('{}{}-{}'.format(mode, i_episode, i), *val_selected)

        if args.model in args.imgxform_models or args.model in args.drastic_models:        
            val_selected = val_selected[1:]
    ########################################################################
    avg_val_loss = np.mean(val_losses)
    avg_val_moves = np.mean(val_moves)
    avg_val_rewards = np.mean(val_rewards)
    avg_val_accuracy = float(sum(val_accuracy)) / args.numval
    trace = 'Trace: {}'.format(val_trace)
    val_correct_stdout = 'CORRECT!' if val_correct == 1 else ''
    stdout = 'Avg Loss: {:.7f}\n\tAvg Moves: {}\n\tAvg Reward: {}\n\tAvg Accuracy: {}\n\tPrediction: {}\t Answer: {}\t\t\t{}\n\tSelected: {}\n\t{}'.format(
        avg_val_loss, avg_val_moves, avg_val_rewards, avg_val_accuracy, val_pred, val_target, val_correct_stdout, val_selected, trace)
    return avg_val_loss, avg_val_moves, avg_val_rewards, avg_val_accuracy, stdout, visualize_data

def eval(data_sampler, episode_sampler, agent, logger, env, args, i_episode):
    base_eval(data_sampler, episode_sampler, validate, agent, logger, env, args)
    ext = '_eval'
    logger.save(logger.expname+ext)

def train(data_sampler, episode_sampler, agent, logger, env, args):
    handle_resume(agent, args)

    run_avg = RunningAverage()
    initialize_logger(logger)
    running_forward_duration = 0
    running_backward_duration = 0
    
    if args.cuda: agent.cuda()
    env.initialize_printer(logger, args)
    env.initialize_data({'train': 0.7, 'val': 0.15, 'test': 0.15})

    ext = '_transfer' if args.transfer else '_train'

    for i_episode in count(0):
        ###############################################################
        # train
        forward_start = time.time()
        loss, selected, pred, target, correct, episode_data = data_sampler(
            episode_sampler=episode_sampler, 
            create_batch=create_batch_transforms,
            bsize=args.bsize,
            agent=agent, 
            env=env, 
            args=args, 
            mode='train')
        moves = len(episode_data)
        reward = sum([e['reward'] for e in episode_data])
        running_loss = run_avg.update_variable('loss', loss)
        running_moves = run_avg.update_variable('moves', moves)
        running_reward = run_avg.update_variable('reward', reward)
        running_accuracy = run_avg.update_variable('accuracy', correct)
        forward_end = time.time()
        forward_duration = forward_end - forward_end
        running_forward_duration = run_avg.update_variable('forward_pass', forward_duration)
        train_trace = env.get_trace()

        if args.model in args.imgxform_models or args.model in args.drastic_models:
            if i_episode % (10*args.val_every) in range(10):
                logger.visualize_transformations('{}{}'.format('train', i_episode), *selected)
            selected = selected[1:]  # just get actions

        ###############################################################
        # stdout
        if i_episode % args.log_interval == 0:
            loss_data = (running_loss, loss)
            moves_data = (running_moves, moves)
            reward_data = (running_reward, reward)
            trace = 'Trace: {}'.format(train_trace)
            correct_stdout = 'CORRECT!' if correct == 1 else ''
            stdout = 'Episode:{}\n\tRunning/Episode Loss: {:.7f}/{:.7f}\n\tRunning/Episode Moves: {}/{}\n\tRunning/Episode Reward: {}/{}\n\tRunning Accuracy: {}\n\tPrediction: {}\tAnswer: {}\t\t\t{}\n\tSelected: {}\n\t{}'.format(
                i_episode, loss_data[0], loss_data[1], moves_data[0], moves_data[1], reward_data[0], reward_data[1], running_accuracy, pred, target, correct_stdout, selected, trace)
            printf(logger, args, stdout)
            printf(logger, args, 'Running Forward Duration: {}'.format(running_forward_duration))
            printf(logger, args, 'Running Backard Duration: {}'.format(running_backward_duration))

        ###############################################################
        # val
        if i_episode % args.val_every == 0:
            avg_val_loss, avg_val_moves, avg_val_rewards, avg_val_accuracy, val_stdout, val_visualize_data = validate(data_sampler, episode_sampler, agent, env, args, logger, mode='val', i_episode=i_episode)
            stdout = 'Validation Episode {}\n\t{}'.format(i_episode, val_stdout)
            printf(logger, args, stdout)
            avg_test_loss, avg_test_moves, avg_test_rewards, avg_test_accuracy, test_stdout, test_visualize_data = validate(data_sampler, episode_sampler, agent, env, args, logger, mode='test', i_episode=i_episode)
            stdout = 'Test Episode {}\n\t{}'.format(i_episode, test_stdout)
            printf(logger, args, stdout)

        ###############################################################
        # update MNIST
        retain_graph = args.memory
        # this is equivalent to what was before if computation_update == policy_update
        should_update_policy = i_episode > 0 and i_episode % args.policy_update == args.policy_update_offset
        should_update_comp = i_episode > 0 and i_episode % args.computation_update == args.computation_update_offset
        should_update_program = True
        should_update_encoder = True

        def update_optimizer_lr(optimizer, scheduler, name):
            before_lr = optimizer.state_dict()['param_groups'][0]['lr']
            scheduler.step()
            after_lr = optimizer.state_dict()['param_groups'][0]['lr']
            to_print_alr = '\nLearning rate for {} was {}. Now it is {}.'.format(name, before_lr, after_lr)
            if before_lr != after_lr:
                to_print_alr += ' Learning rate changed!\n'
                printf(logger, args, to_print_alr)

        if i_episode >= args.anneal_policy_lr_after:
            update_optimizer_lr(
                optimizer=agent.policy_optimizer,
                scheduler=agent.po_scheduler,
                name='policy')
            update_optimizer_lr(
                optimizer=agent.value_optimizer,
                scheduler=agent.vo_scheduler,
                name='value')

        if i_episode >= args.anneal_comp_lr_after:
            update_optimizer_lr(
                optimizer=agent.computation_optimizer, 
                scheduler=agent.co_scheduler,
                name='computation')

        if should_update_policy:

            backward_start = time.time()
            agent.improve_policy_ppo()
            backward_end = time.time()
            backward_duration = backward_end - backward_start
            running_backward_duration = run_avg.update_variable('backward_pass', backward_duration)
            agent.replay_buffer.clear_buffer()

        if should_update_comp:
            agent.improve_actions(retain_graph=retain_graph)
            agent.computation_buffer.clear_buffer()

        ###############################################################
        # log
        if i_episode % args.save_every == 0:

            logger.update_variable('episode', i_episode)
            logger.update_variable('loss', loss)
            logger.update_variable('running_loss', running_loss)
            logger.update_variable('moves', moves)
            logger.update_variable('running_moves', running_moves)
            logger.update_variable('reward', reward)
            logger.update_variable('running_reward', running_reward)
            logger.update_variable('running_accuracy', running_accuracy)
            logger.update_variable('val_loss', avg_val_loss)
            logger.update_variable('val_moves', avg_val_moves)
            logger.update_variable('val_reward', avg_val_rewards)
            logger.update_variable('val_accuracy', avg_val_accuracy)
            logger.update_variable('test_loss', avg_test_loss)
            logger.update_variable('test_moves', avg_test_moves)
            logger.update_variable('test_reward', avg_test_rewards)
            logger.update_variable('test_accuracy', avg_test_accuracy)

            logger.update_variable('forward_pass', running_forward_duration)
            logger.update_variable('backward_pass', running_backward_duration)

            logger.save(logger.expname+ext)

            # save a checkpoint
            current_metrics = {
                'val_loss': avg_val_loss, 
                'val_moves': avg_val_moves, 
                'val_reward': avg_val_rewards, 
                'val_accuracy': avg_val_accuracy, 
                'running_loss': running_loss, 
                'running_accuracy': running_accuracy}
            ckpt = {
                'model': {k: to_cpu(v.state_dict()) for k,v in agent.model.iteritems()},
                'optimizer': {k: v.state_dict() for k,v in agent.optimizer.iteritems()},
                'episode': i_episode,
                'running_loss': running_loss,
                'running_moves': running_moves,
                'running_reward': running_reward,
                'running_accuracy': running_accuracy,
                'val_loss': avg_val_loss,
                'val_moves': avg_val_moves,
                'val_reward': avg_val_rewards,
                'val_accuracy': avg_val_accuracy,
                'test_loss': avg_test_loss,
                'test_moves': avg_test_moves,
                'test_reward': avg_test_rewards,
                'test_accuracy': avg_test_accuracy,
                'forward_pass': running_forward_duration,
                'backward_pass': running_backward_duration,
                'logger_data': logger.data,
                'resumed_from': logger.resumed_from
            }

            logger.save_checkpoint(ckpt, current_metrics, i_episode, args, ext)

        ###############################################################
        # plot
        if i_episode % args.plot_every == 0:
            base_plot(logger)

        ###############################################################
        # update curriculum
        if i_episode % args.curr_every == 0 and args.curr and i_episode > 0:
            printf(logger, args, '*'*80)
            printf(logger, args, 'Update Curriculum')
            printf(logger, args, '*'*80)
            env.update_curriculum()

        ##############################################################
        if i_episode >= args.max_episodes:
            printf(logger, args, 'Training done for {}'.format(logger.logdir))
            break

