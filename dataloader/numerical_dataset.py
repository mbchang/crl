import torch
from torchvision import datasets, transforms
from collections import OrderedDict
import numpy as np
import copy
import os
import re

from arithmetic import Plus, Minus, Multiply, Divide
import datautils as du
from tqdm import tqdm
from modulo_datagen import ModuloDataGenerator
from utils import printf

def mkdirp(logdir):
    if not os.path.exists(logdir):
        os.mkdir(logdir)

class DataSplitter(object):
    def __init__(self, train_split, val_split, test_split, printer):
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.printer = printer

    def create_dataset_splits(self, datagen, cheat=False):
        problems = list(datagen())
        num_total = len(problems)
        num_test = int(np.ceil(self.test_split*num_total))
        num_val = int(np.ceil(self.val_split*num_total))
        num_train = num_total - num_test - num_val
        if cheat:
            train_problems = copy.deepcopy(problems)
        else:
            train_problems = problems[:num_train]
        val_problems = problems[num_train:num_train+num_val]
        test_problems = problems[num_train+num_val:]
        self.printer('Total Problems: {} Train Problems: {} Val Problems: {} Test Problems: {}'.format(
            num_total, len(train_problems), len(val_problems), len(test_problems)))
        return {'train': train_problems, 'val': val_problems, 'test': test_problems}

def sort_ops(ops):
    sorted_ops = ''
    if '+' in ops:
        sorted_ops+='+'
    if '*' in ops:
        sorted_ops+='*'
    if '-' in ops:
        sorted_ops+='-'
    if '/' in ops:
        sorted_ops+='/'
    assert len(sorted_ops) > 0
    return sorted_ops

# this is responsible for creating math expressions, saving, and loading datasets
class BaseArithmetic(object):
    def __init__(self, max_terms, num_range, ops, samplefrom, episodecap, root, curr):
        self.printer = self.initialize_printer(None, None)  # YES
        self.operators = list(ops)
        self.num_range = num_range
        self.cheat = False
        self.curr = curr

        self.max_terms = max_terms
        self.max_terms_dict = {'train': max_terms[0], 'val': max_terms[1], 'test': max_terms[1]}

        self.dg = ModuloDataGenerator(ops, num_range)

        self.ops = sort_ops(ops)

        self.samplefrom = samplefrom
        self.episodecap = episodecap
        self.root = root

        self.should_change_mt = True

    def initialize_printer(self, logger, args):
        if logger is None:
            assert args is None
            def printer(x):
                print x
        else:
            printer = lambda x: printf(logger, args, x)
        self.printer = printer

    def initialize_data(self, splits):
        if self.curr:
            self.initial_mt = 2
            assert self.max_terms[0] == self.max_terms[1]
            self.curriculum_schedule = range(self.initial_mt, self.max_terms_dict['train']+1)  # should change later
        else:
            assert self.max_terms[0] == self.max_terms[1]
            self.initial_mt = self.max_terms_dict['train']
            self.curriculum_schedule = [self.max_terms_dict['train'], self.max_terms_dict['val'], self.max_terms_dict['test']]
        self.extrapval_mt = self.max_terms[2]

        self.ext = '.txt'
        mkdirp(self.root)
        self.ds = DataSplitter(train_split=0.7, val_split=0.15, test_split=0.15, printer=self.printer)
        mts = set(self.curriculum_schedule+[self.extrapval_mt])
        self.dataset_names = self.save_all_datasets(mts, self.num_range, self.ops, self.samplefrom, self.episodecap, self.cheat, splits)
        self.initialize_problems()
        self.get_problem = self.load_problem

    def save_all_datasets(self, mts, num_range, ops, samplefrom, episodecap, cheat, splits):
        dataset_names = {}
        for mt in mts:  
            dataset_name = self.create_dataset_name(mt, num_range, ops, samplefrom, episodecap, cheat, splits)
            mkdirp(os.path.join(self.root, dataset_name))
            dataset_names[mt] = {x: os.path.join(self.root, dataset_name, dataset_name + '_' + x + '.txt') for x in ['train', 'val', 'test']}
            datasets_exist = all([os.path.exists(v) for k,v in dataset_names[mt].iteritems()])
            self.printer('datasets exist for {}? {}'.format(mt, datasets_exist))
            if not datasets_exist:
                self.printer('Datasets for {} are not found! Generating data.'.format(os.path.join(self.root, dataset_name)))
                self.save_datasets(dataset_names[mt], self.samplefrom, self.episodecap, mt, False)
        return dataset_names

    def initialize_problems(self):
        self.curriculum_problems = {}
        self.curriculum_problem_counters = {}
        self.curriculum_counters = {'train': 0, 'val': 0, 'test': 0}

        self.add_dataset('train', self.initial_mt)
        self.add_dataset('val', self.initial_mt)
        self.add_dataset('test', self.initial_mt)

        dataset_name_for_extrapolation = self.get_dataset_name_for_mode_k('val', self.extrapval_mt)
        self.extrapolation_problems = self.load_dataset(dataset_name_for_extrapolation)
        self.extrapolation_counter = 0
        self.printer('Adding {} data from {} to dataset'.format('val', dataset_name_for_extrapolation))

    def create_dataset_name(self, max_terms, num_range, ops, samplefrom, episodecap, cheat, splits):
        dataset_name = 'arith_mt{}_r{}_o{}_from{}_cap{}_split{}-{}-{}'.format(
            max_terms, str(num_range).replace(' ',''), ops.replace('/','d'), int(samplefrom), int(episodecap),
            splits['train'], splits['val'], splits['test'])
        dataset_name += '_mod'
        if cheat: dataset_name += '_cheat'
        return dataset_name

    def extract_terms_ops(self, exp_str):
        operator_matcher = re.compile(r'[\+\-\*\/]')
        operators = OrderedDict([
                    ('+', Plus),
                    ('*', Multiply),
                    ('-', Minus),
                    ('/', Divide)
                    ])
        return du.extract_terms_ops(exp_str, operator_matcher, operators)

    def load_problem(self, mode, mt):
        if mode == 'extrapval':
            if self.extrapolation_counter >= len(self.extrapolation_problems):
                np.random.shuffle(self.extrapolation_problems)
                self.extrapolation_counter = 0
            problem = self.extrapolation_problems[self.extrapolation_counter]
            self.extrapolation_counter += 1
        else:
            if self.curriculum_problem_counters[mode][mt] >= len(self.curriculum_problems[mode][mt]):
                np.random.shuffle(self.curriculum_problems[mode][mt])
                self.curriculum_problem_counters[mode][mt] = 0
            problem = self.curriculum_problems[mode][mt][self.curriculum_problem_counters[mode][mt]]
            self.curriculum_problem_counters[mode][mt] += 1

        # convert it to standard form
        exp_str = problem[:problem.find('=')]
        exp_val = int(problem[problem.find('=')+1:])
        terms, ops = self.extract_terms_ops(exp_str)
        return exp_str, exp_val, terms, ops

    def get_dataset_name_for_mode_k(self, mode, k):
        return self.dataset_names[k][mode]

    # you should have a method that adds a dataset to the problems
    def add_dataset(self, mode, k):
        """
        adds a dataset with k terms into the datasets
        """ 
        if mode not in self.curriculum_problems:
            assert mode not in self.curriculum_problem_counters
            self.curriculum_problems[mode] = {}
            self.curriculum_problem_counters[mode] = {}
        if k not in self.curriculum_problems[mode]:
            assert k not in self.curriculum_problem_counters[mode]
            dataset_name_for_mode_k = self.get_dataset_name_for_mode_k(mode, k)
            self.printer('Adding {} data from {} to dataset'.format(mode, dataset_name_for_mode_k))
            self.curriculum_problems[mode][k] = self.load_dataset(dataset_name_for_mode_k)
            self.curriculum_problem_counters[mode][k] = 0
        else:
            assert False, 'Dataset for {} {} is already added'.format(mode, k)

    def delete_dataset(self, mode, k):
        del self.curriculum_problems[mode][k]
        del self.curriculum_problem_counters[mode][k]

    def update_curriculum(self):
        for mode in ['train', 'val', 'test']:
            if self.curriculum_counters[mode] < len(self.curriculum_schedule)-1:
                self.curriculum_counters[mode] += 1
                mt = self.curriculum_schedule[self.curriculum_counters[mode]]
                self.add_dataset(mode, mt)
                if mode in ['val', 'test']:
                    self.delete_dataset(mode, self.curriculum_schedule[self.curriculum_counters[mode]-1])

    def generate_unique_dataset(self, max_problems, cap, maxterms):
        problems = set()
        for i in tqdm(xrange(int(max_problems))):
            exp_str, exp_val, terms, ops = self.dg.create_problem(max_terms=maxterms)
            problems.add('{}={}'.format(exp_str, exp_val))
        # cap it at some max value
        problems = list(problems)[:cap]
        return problems

    def save_datasets(self, dataset_names, samplefrom, episodecap, maxterms):
        problems = self.ds.create_dataset_splits(lambda: self.generate_unique_dataset(samplefrom, int(episodecap), maxterms), self.cheat)
        i = 0
        while(any(self.insufficient_coverage(problems[k]) for k in problems)):
            np.random.seed(i)
            self.printer('Insufficient coverage! Trying again.')
            problems = self.ds.create_dataset_splits(lambda: self.generate_unique_dataset(samplefrom, int(episodecap), maxterms), self.cheat)
            i += 1
        for k,v in dataset_names.iteritems():
            self.printer('Saving {}'.format(v))
            self.save_dataset(problems[k], v)

    def insufficient_coverage(self, problems):
        """
            check that all terms are covered
            check that all ops are covered

            returns True if insufficient coverage of dataset
        """
        terms = set()
        ops = set()
        for p in problems:
            t, o = self.extract_terms_ops(p[:p.find('=')])
            terms.update(t)
            ops.update(o)
        insufficient_ops = len(ops) < len(self.operators)
        insufficient_terms = len(set(range(min(self.num_range), max(self.num_range))).difference(terms)) > 0
        return insufficient_ops or insufficient_terms

    def save_dataset(self, problems, fname):
        # we use 'a' because we won't save the dataset if the file exists already.
        with open(fname, 'a') as f:
            for p in problems:
                f.write(p+'\n')

    def load_dataset(self, fname):
        with open(fname, 'r') as f:
            problems = f.readlines()
        return problems

    def reset(self, mode): 
        if mode == 'train':
            if self.should_change_mt:
                self.mt = np.random.choice(self.curriculum_schedule[:self.curriculum_counters[mode]+1])
                self.should_change_mt = False
            mt = self.mt
        elif mode == 'val' or mode == 'test':
            mt = self.curriculum_schedule[self.curriculum_counters[mode]]
        elif mode == 'extrapval':
            mt = self.extrapval_mt
        else:
            assert False
        exp_str, exp_val, terms, ops = self.get_problem(mode, mt)
        return exp_str, exp_val, terms, ops

    def change_mt(self):
        self.should_change_mt = True
