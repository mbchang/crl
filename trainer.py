import numpy as np
import operator

def to_cpu(state_dict):
    cpu_dict = {}
    for k,v in state_dict.iteritems():
        cpu_dict[k] = v.cpu()
    return cpu_dict

def initialize_logger(logger):
    logger.add_variables([
        'episode', 
        'loss', 'moves', 'reward', 'accuracy',
        'running_loss', 'running_moves', 'running_reward', 'running_accuracy',
        'val_loss', 'val_moves', 'val_reward', 'val_accuracy', 
        'test_loss', 'test_moves', 'test_reward', 'test_accuracy',
        'extrapval_loss', 'extrapval_moves', 'extrapval_reward', 'extrapval_accuracy',
        'forward_pass', 'backward_pass',
        'val_hist', 'test_hist', 'extrap_hist',
        ])

    logger.add_variable('forward_pass')
    logger.add_variable('backward_pass')

    logger.add_metric('val_loss', np.inf, operator.le)
    logger.add_metric('val_moves', np.inf, operator.le)
    logger.add_metric('val_reward', -np.inf, operator.ge)
    logger.add_metric('val_accuracy', -np.inf, operator.ge)

    logger.add_metric('running_loss', np.inf, operator.le)
    logger.add_metric('running_accuracy', -np.inf, operator.ge)

    logger.add_unique_sets(['train', 'val', 'test', 'extrapval'])


def base_eval(data_sampler, episode_sampler, validate, agent, logger, env, args):
    handle_resume(agent, args)
    initialize_logger(logger)
    if args.cuda: agent.cuda()
    env.initialize_printer(logger, args)
    env.initialize_data({'train': 0.7, 'val': 0.15, 'test': 0.15})
    i_episode = 0
    avg_val_loss, avg_val_moves, avg_val_rewards, avg_val_accuracy, val_stdout, val_visualize_data = validate(data_sampler, episode_sampler, agent, env, args, logger, mode='val', i_episode=i_episode)
    stdout = 'Validation Episode {}\n\t{}'.format(i_episode, val_stdout)
    printf(logger, args, stdout)
    avg_test_loss, avg_test_moves, avg_test_rewards, avg_test_accuracy, test_stdout, test_visualize_data = validate(data_sampler, episode_sampler, agent, env, args, logger, mode='test', i_episode=i_episode)
    stdout = 'Test Episode {}\n\t{}'.format(i_episode, test_stdout)
    printf(logger, args, stdout)
    if isinstance(env, BaseArithmetic) and env.extrapval_mt > env.max_terms_dict['val']:
        avg_extrapval_loss, avg_extrapval_moves, avg_extrapval_rewards, avg_extrapval_accuracy, extrapval_stdout, extrapval_visualize_data = validate(
            data_sampler, 
            episode_sampler, agent, env, args, logger, mode='extrapval')
        stdout = 'Extrapolation Validation Episode {}\n\t{}'.format(i_episode, extrapval_stdout)
        printf(logger, args, stdout)
    else:
        printf(logger, args, 'Did not do extrapolation test because extrapolation dataset is not given.')

    # ###############################################################
    # log
    logger.update_variable('val_loss', avg_val_loss)
    logger.update_variable('val_moves', avg_val_moves)
    logger.update_variable('val_reward', avg_val_rewards)
    logger.update_variable('val_accuracy', avg_val_accuracy)
    logger.update_variable('test_loss', avg_test_loss)
    logger.update_variable('test_moves', avg_test_moves)
    logger.update_variable('test_reward', avg_test_rewards)
    logger.update_variable('test_accuracy', avg_test_accuracy)


def base_plot(logger):
    logger.plot('episode', 'loss', logger.expname+'_loss')
    logger.plot('episode', 'running_loss', logger.expname+'_running_loss')
    logger.plot('episode', 'moves', logger.expname+'_moves')
    logger.plot('episode', 'running_moves', logger.expname+'_running_moves')
    logger.plot('episode', 'reward', logger.expname+'_reward')
    logger.plot('episode', 'running_reward', logger.expname+'_running_reward')
    logger.plot('episode', 'running_accuracy', logger.expname+'_running_accuracy')
    logger.plot('episode', 'val_loss', logger.expname + '_val_loss')
    logger.plot('episode', 'val_moves', logger.expname + '_val_moves')
    logger.plot('episode', 'val_reward', logger.expname + '_val_reward')
    logger.plot('episode', 'val_accuracy', logger.expname + '_val_accuracy')
    logger.plot('episode', 'test_loss', logger.expname + '_test_loss')
    logger.plot('episode', 'test_moves', logger.expname + '_test_moves')
    logger.plot('episode', 'test_reward', logger.expname + '_test_reward')
    logger.plot('episode', 'test_accuracy', logger.expname + '_test_accuracy')
    logger.plot('episode', 'forward_pass', logger.expname + '_forward_pass')
    logger.plot('episode', 'backward_pass', logger.expname + '_backward_pass')
