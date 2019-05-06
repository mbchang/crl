import os
import pickle
import torch
import copy
from utils import printf

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from dataloader.image_transforms import convert_image_np

def mkdirp(logdir):
    if '_debug' in logdir:
        # overwrite
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    else:
        os.mkdir(logdir)

def barplot(height, labels, fname):
    N = len(height)
    x = np.array(range(N))
    plt.bar(x=x, height=height, align='center')
    plt.xticks(x, labels, rotation='vertical')
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*2*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

class Logger(object):
    def __init__(self, expname, logdir, params, variables=None, resumed_from=None):
        super(Logger, self).__init__()
        self.data = {}
        self.metrics = {}

        self.unique_problems = {}

        self.expname = expname
        self.logdir = logdir
        self.params = params
        self.resumed_from = resumed_from if resumed_from else None

        if self.resumed_from:
            assert os.path.exists(self.resumed_from)
            if os.path.dirname(self.resumed_from) != self.logdir:
                mkdirp(self.logdir)
        else:
            mkdirp(self.logdir)

        if variables is not None:
            self.add_variables(variables)

    def set_expname(self, expname):
        self.expname = expname

    def set_resumed_from(self, resume_path):
        self.data['resumed_from'] = resume_path

    #############################################
    def load_params_eval(self, eval_, resume):
        """ saved_args is mutable! """
        assert self.resumed_from is not None
        saved_args = self.load_params()
        saved_args.eval = eval_
        saved_args.resume = resume
        self.set_resumed_from(self.resumed_from)
        return saved_args


    def load_params_transfer(self, transfer, resume):
        """ saved_args is mutable! """
        assert self.resumed_from is not None
        saved_args = self.load_params()
        saved_args.transfer = transfer
        saved_args.resume = resume
        self.set_resumed_from(self.resumed_from)
        return saved_args

    # should be able to combine the above
    #############################################

    def save_params(self, logdir, params, ext=''):
        pickle.dump(params, open(os.path.join(self.logdir, '{}.p'.format('params'+ext)), 'wb'))

    def set_params(self, params):
        """ params is mutable """
        self.params = params

    def set_and_save_params(self, logdir, params, ext=''):
        self.set_params(params)
        self.save_params(logdir, params, ext)

    def load_params(self):
        params = pickle.load(open(os.path.join(self.logdir, '{}.p'.format('params')), 'rb'))
        return params

    def add_variables(self, names):
        for name in names:
            self.add_variable(name)

    def update_variables(self, name_values):
        for name, value in name_values:
            self.update_variable(name, value)

    def add_variable(self, name):
        self.data[name] = []

    def update_variable(self, name, value):
        self.data[name].append(value)

    def add_metric(self, name, initial_val, comparator):
        self.metrics[name] = {'value': initial_val, 'cmp': comparator}

    def add_unique_sets(self, names):
        for name in names:
            self.add_unique_set(name)

    def add_unique_set(self, name):
        self.unique_problems[name] = set()

    def update_unique_set(self, name, key):
        self.unique_problems[name].add(key)

    def get_unique_set_size(self, name):
        return len(self.unique_problems[name])

    def save_checkpoint(self, ckpt_data, current_metrics, i_episode, args, ext):
        old_ckpts = [x for x in os.listdir(self.logdir) if '.pth.tar' in x and 'best' in x and ext in x]
        assert len(old_ckpts) <= len(current_metrics)

        for m in self.metrics:
            if self.metrics[m]['cmp'](current_metrics[m], self.metrics[m]['value']):
                self.metrics[m]['value'] = current_metrics[m]
                if any(m in oc for oc in old_ckpts):
                    old_ckpts_with_metric = [x for x in old_ckpts if 'best{}'.format(m) in x]
                    assert len(old_ckpts_with_metric) == 1
                    old_ckpt_to_remove = os.path.join(self.logdir,old_ckpts_with_metric[0])
                    os.remove(old_ckpt_to_remove)
                    printf(self, args, 'Removing {}'.format(old_ckpt_to_remove))
                torch.save(ckpt_data, os.path.join(self.logdir, '{}_ep{:.0e}_best{}{}.pth.tar'.format(
                    self.expname, i_episode, m, ext)))
                printf(self, args, 'Saved Checkpoint for best {}'.format(m))
            else:
                printf(self, args, 'Did not save {} checkpoint at because {} was worse than the best'.format(m, m))

    def plot(self, var1, var2, fname):
        plt.plot(self.data[var1], self.data[var2])
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.savefig(os.path.join(self.logdir,'{}.png'.format(fname)))
        plt.clf()

    def add_variable_hist(self, name, bins):
        self.data[name+'_hist'] = {'values': [], 'bins': bins}

    def update_variable_hist(self, name, value):
        self.data[name+'_hist']['values'].append(value)

    def plot_hist(self, name, fname):
        plt.hist(self.data[name+'_hist']['values'], bins=self.data[name+'_hist']['bins'])
        plt.savefig(os.path.join(self.logdir,'{}.png'.format(fname)))
        plt.close()

    def add_variable_bar(self, name, num_bars, labels):
        self.data[str(name)+'_bar'] = {'values': [0 for j in range(num_bars)], 'labels': labels}
    
    def increment_variable_bar(self, name, idx, incr):
        self.data[str(name)+'_bar']['values'][idx] += incr

    def plot_bar(self, name, fname):
        barplot(height=np.array(self.data[str(name)+'_bar']['values']), 
                labels=self.data[str(name)+'_bar']['labels'],
                fname=os.path.join(self.logdir,'{}.png'.format(fname)))

    def to_cpu(self, state_dict):
        cpu_dict = {}
        for k,v in state_dict.iteritems():
            cpu_dict[k] = v.cpu()
        return cpu_dict

    def saveckpt(self, filename, ckpt):
        save_path = os.path.join(self.logdir, filename)
        state = {

                # 'model': {k: v.state_dict() for k,v in agent.model.iteritems()},
                # 'optimizer': {k: v.state_dict() for k,v in agent.optimizer.iteritems()},

            'model': {k: self.to_cpu(v) for k,v in ckpt['model'].iteritems()},
            'episode': ckpt['episode'],
            'running_reward': ckpt['running_reward'],
            'logger_data': ckpt['logger_data'],
            'resumed_from': self.resumed_from
        }
        if type(ckpt['optimizer']) is list:
            state['optimizer'] = [o.state_dict() for o in ckpt['optimizer']]
        else:
            state['optimizer'] = ckpt['optimizer'].state_dict()
        torch.save(state, save_path)
        return save_path

    def save(self, name):
        pickle.dump(self.data, open(os.path.join(self.logdir,'{}.p'.format(name)), 'wb'))

    def load(self, name):
        self.data = pickle.load(open(os.path.join(self.logdir,'{}.p'.format(name)), 'rb'))

    def visualize_transformations(self, fname, selected_states, selected_actions, visualize=False):
        states_np = map(lambda x: x[0], map(convert_image_np, map(lambda x: x.cpu(), selected_states)))
        f, ax = plt.subplots(1, len(states_np))
        for i in range(len(states_np)):
            ax[i].imshow(states_np[i])
            if i > 0:
                ax[i].set_title('After action {}'.format(selected_actions[i-1]), fontsize=10)
        plt.savefig(os.path.join(self.logdir, '{}.png'.format(fname)))
        plt.close()


# You can make this subclass a base AccuracyTracker
class AccuracyTracker(object):
    def __init__(self, keys):
        self.keys = keys
        self.agg = lambda x: x.long().cpu().sum()
        self.reset()

    def reset(self):
        self.corrects = {x: 0 for x in self.keys}
        self.totals = {x: 0 for x in self.keys}

    def accumulate_multi_target(self, pred, target, subvocabsize, nlang):
        """
            pred: (target_bsize, 1)
            target: (target_bsize,)

            possible_targets: (target_bsize, nlang)
            matches: (target_bsize, nlang)
            anyequal: (target_bsize,)

            filtered_target: (target_bsize,)
        """
        possible_targets = []
        for i in range(nlang):
            possible_targets.append(target+i*subvocabsize)
        possible_targets = torch.stack(possible_targets, dim=1)
        matches = pred.repeat(1,nlang).eq(possible_targets)
        anyequal = (matches.sum(1) != 0)
        correct = 0
        for k in self.keys:
            filtered_target = target.eq(k)
            filtered_match = self.agg(anyequal * filtered_target)
            correct += filtered_match
            self.corrects[k] += filtered_match
            self.totals[k] += self.agg(filtered_target)
        return correct

    def accumulate_single_target(self, pred, target):
        """
            pred: (target_bsize, 1)
            target: (target_bsize,)

            # assume that each element in pred and target are in self.keys

            # if you do not do it by key, then this is just
            pred.eq(target.view_as(pred)).long().cpu().sum()
        """
        target = target.view_as(pred)
        matches = pred.eq(target)
        correct = 0
        for k in self.keys:
            filtered_target = target.eq(k)
            filtered_match = self.agg(matches * filtered_target)
            correct += filtered_match
            self.corrects[k] += filtered_match
            self.totals[k] += self.agg(filtered_target)
        return correct

    def get_correct(self, k):
        return self.corrects[k]

    def get_total(self, k):
        return self.totals[k]

    def get_accuracy(self, k):
        if self.totals[k] > 0:
            return 100. * self.corrects[k]/self.totals[k]
        else:
            return 0

    def print_stats(self, mode, subvocabsize):
        for i in range(subvocabsize):
            print('{} {}: {}/{} ({:.2f}%)'.format(mode, i,
                self.get_correct(i),
                self.get_total(i),
                self.get_accuracy(i)))


class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
        if 'running_'+key not in self.data:
            self.data['running_'+key] = value
        else:
            self.data['running_'+key] = (1-self.alpha) * self.data['running_'+key] + self.alpha * value
        return copy.deepcopy(self.data['running_'+key])

    def get_value(self, key):
        if key in self.data:
            return self.data['running_'+key]
        else:
            assert KeyError

def visualize_parameters(model):
    for n, p in model.named_parameters():
        if p.grad is None:
            print n, p.data.norm(), None
        else:
            print n, p.data.norm(), p.grad.data.norm()

def count_params(model):
    print 'Total Paramters {}'.format(
        sum(p.numel() for p in model.parameters()))
    print 'Trainable Parameters {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad))


