import itertools
from itertools import count
import numpy as np
import os
import operator
import shutil
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import dataloader.datautils as du
from dataloader.numerical_dataset import BaseArithmetic
from load_pretrained import handle_resume
from log import visualize_parameters, RunningAverage
from utils import cuda_if_needed, printf
from trainer import initialize_logger, to_cpu, base_eval, base_plot

def create_lang_batch(env, bsize, mode, args):
    volatile = mode != 'train'
    z = 1
    whole_expr = np.random.binomial(n=1, p=0.5)
    enc_inps = []
    target_tokens = []
    zs = []
    targets = []

    for j in range(bsize):
        initial, target = env.reset(mode, z)
        enc_inps.append(np.stack([du.num2onehot(x, env.vocabsize) for x in initial[0]]))

        target_tokens.append(du.num2onehot(initial[1], env.langsize))
        zs.append(du.num2onehot(initial[2], env.zsize))
        targets.append(target)

    env.change_mt()

    enc_inps = torch.FloatTensor(np.array(enc_inps))  # (b, inp_seq_length, vocabsize)

    target_tokens = torch.FloatTensor(target_tokens)  # (b, langsize)
    zs = torch.FloatTensor(zs)  # (b, zsize)

    targets = torch.LongTensor(targets)  # (b, 1)

    enc_inps, target_tokens, zs, targets = map(lambda x: cuda_if_needed(x, args), 
        (enc_inps, target_tokens, zs, targets))

    targets = Variable(targets, volatile=volatile)

    return (enc_inps, target_tokens, zs), targets

def validate(data_sampler, episode_sampler, agent, env, args, logger, mode, i_episode):
    val_losses = []
    val_moves = []
    val_rewards = []
    val_accuracy = []
    visualize_data = []
    for i in range(args.numval):
        val_loss, val_selected, val_pred, val_target, val_correct, val_episode_data = data_sampler(
            episode_sampler, create_lang_batch, args.bsize, agent, env, args, mode=mode)
        val_losses.append(val_loss)
        val_moves.append(len(val_episode_data))
        val_rewards.append(sum([e['reward'] for e in val_episode_data]))
        val_accuracy.append(val_correct)
        val_trace = env.get_trace()

        ############################################################
        # 2: STOP 1: REDUCE 0: TRANSLATE
        meta_selected = agent.get_meta_selected(val_selected)
        sub_selected = agent.get_sub_selected(val_selected)
        assert len(env.current_exp_strs) == len(val_selected) + 1
        visualize_data.append({
            'trace': env.current_exp_strs,
            'meta_selected': meta_selected,
            'sub_selected': sub_selected,
            'probs': [v['choice_dist_info'] for v in val_episode_data],
            'moves': len(val_selected),
            'correct': val_correct,
            }
        )
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

def eval(data_sampler, episode_sampler, agent, logger, env, args):
    base_eval(data_sampler, episode_sampler, validate, agent, logger, env, args)
    if isinstance(env, BaseArithmetic) and env.extrapval_mt > env.max_terms_dict['val']:
        logger.update_variable('extrapval_loss', avg_extrapval_loss)
        logger.update_variable('extrapval_moves', avg_extrapval_moves)
        logger.update_variable('extrapval_reward', avg_extrapval_rewards)
        logger.update_variable('extrapval_accuracy', avg_extrapval_accuracy)
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

    ext = '_train'

    for i_episode in count(1):
        ###############################################################
        # train
        forward_start = time.time()
        loss, selected, pred, target, correct, episode_data = data_sampler(
            episode_sampler=episode_sampler, 
            create_batch=create_lang_batch,
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

        train_trace = env.get_trace()

        forward_end = time.time()
        forward_duration = forward_end - forward_end
        running_forward_duration = run_avg.update_variable('forward_pass', forward_duration)

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
        if i_episode % args.val_every == 0 or i_episode == 1:
            avg_val_loss, avg_val_moves, avg_val_rewards, avg_val_accuracy, val_stdout, val_visualize_data = validate(data_sampler, episode_sampler, agent, env, args, logger, mode='val', i_episode=i_episode)
            stdout = 'Validation Episode {}\n\t{}'.format(i_episode, val_stdout)
            printf(logger, args, stdout)
            avg_test_loss, avg_test_moves, avg_test_rewards, avg_test_accuracy, test_stdout, test_visualize_data = validate(data_sampler, episode_sampler, agent, env, args, logger, mode='test', i_episode=i_episode)
            stdout = 'Test Episode {}\n\t{}'.format(i_episode, test_stdout)
            printf(logger, args, stdout)

            if isinstance(env, BaseArithmetic) and env.extrapval_mt > env.max_terms_dict['val']:
                avg_extrapval_loss, avg_extrapval_moves, avg_extrapval_rewards, avg_extrapval_accuracy, extrapval_stdout, extrapval_visualize_data = validate(
                    data_sampler, 
                    episode_sampler, agent, env, args, logger, mode='extrapval', i_episode=i_episode)
                stdout = 'Extrapolation Validation Episode {}\n\t{}'.format(i_episode, extrapval_stdout)
                printf(logger, args, stdout)
            else:
                printf(logger, args, 'Did not do extrapolation test because extrapolation dataset is not given.')

        ###############################################################
        # update
        retain_graph = args.memory
        # this is equivalent to what was before if computation_update == policy_update
        should_update_policy = i_episode % args.policy_update == args.policy_update_offset
        should_update_comp = i_episode % args.computation_update == args.computation_update_offset

        if should_update_policy:
            agent.po_scheduler.step()
            agent.vo_scheduler.step()
            backward_start = time.time()
            agent.improve_policy_ppo()
            backward_end = time.time()
            backward_duration = backward_end - backward_start
            running_backward_duration = run_avg.update_variable('backward_pass', backward_duration)
            agent.replay_buffer.clear_buffer()

        if should_update_comp:
            agent.co_scheduler.step()
            agent.improve_actions(retain_graph=retain_graph)
            agent.computation_buffer.clear_buffer()
            
        ###############################################################
        # log
        if i_episode % args.save_every == 0 or i_episode == 1:
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

            if isinstance(env, BaseArithmetic) and env.extrapval_mt > env.max_terms_dict['val']:
                logger.update_variable('extrapval_loss', avg_extrapval_loss)
                logger.update_variable('extrapval_moves', avg_extrapval_moves)
                logger.update_variable('extrapval_reward', avg_extrapval_rewards)
                logger.update_variable('extrapval_accuracy', avg_extrapval_accuracy)
            logger.update_variable('forward_pass', running_forward_duration)
            logger.update_variable('backward_pass', running_backward_duration)

            ext = ''
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

            if isinstance(env, BaseArithmetic) and env.extrapval_mt > env.max_terms_dict['val']:
                ckpt.update({
                    'extrapval_loss': avg_extrapval_loss,
                    'extrapval_moves': avg_extrapval_moves,
                    'extrapval_reward': avg_extrapval_rewards,
                    'extrapval_accuracy': avg_extrapval_accuracy,
                    })
            logger.save_checkpoint(ckpt, current_metrics, i_episode, args, ext)

        ###############################################################
        # plot
        if args.plot and i_episode % args.plot_every == 0:
            base_plot(logger)
            if isinstance(env, BaseArithmetic) and env.extrapval_mt > env.max_terms_dict['val']:
                logger.plot('episode', 'extrapval_loss', logger.expname + '_extrapval_loss')
                logger.plot('episode', 'extrapval_moves', logger.expname + '_extrapval_moves')
                logger.plot('episode', 'extrapval_reward', logger.expname + '_extrapval_reward')
                logger.plot('episode', 'extrapval_accuracy', logger.expname + '_extrapval_accuracy')

        ###############################################################
        # update curriculum
        if i_episode % args.curr_every == 0:
            env.update_curriculum()

        ##############################################################
        if i_episode >= args.max_episodes:
            printf(logger, args, 'Training done for {}'.format(logger.logdir))
            break

