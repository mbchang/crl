from itertools import groupby
from operator import itemgetter
import torch
from torch.autograd import Variable
from torch.distributions import Categorical



def printf(logger, args, string):
    if args.printf:
        f = open(logger.logdir+'.txt', 'a')
        print >>f, string
    else:
        print string


def create_exp_string(args, relevant_arg_names, prefix, suffix):
    string = prefix + '_'
    d = vars(args)
    for key in sorted(set(relevant_arg_names)):
        val = d[key]
        to_append = key if isinstance(val, bool) else key + '_' + str(val)
        string += to_append + '_'
    string += suffix
    return string

def inrange(value, interval):
    """
    Outputs whether value > interval[0]
        and < interval[1], inclusive
    """
    # return value >= interval[0] and value <= interval[1]  # NOTE I will change this to exclusive!!
    return value >= interval[0] and value < interval[1]  # NOTE I will change this to exclusive!!


def group_consecutive(list_of_numbers):
    groups = []
    for k, g in groupby(enumerate(list_of_numbers), lambda (i, x): i-x):
        mg = map(itemgetter(1), g)
        groups.append(tuple(mg))
    return groups

def cuda_if_needed(x, args):
    if args.cuda:
        return x.cuda()
    else:
        return x

def group_by_element(list_of_numbers):
    """
    m = [3,3,3,1,2,2,4,4,4,4,5,5,5,5,5]
    idx, vals = group(m)
    -->
    idx = [[0, 1, 2], [3], [4, 5], [6, 7, 8, 9], [10, 11, 12, 13, 14]]
    vals = [[3, 3, 3], [1], [2, 2], [4, 4, 4, 4], [5, 5, 5, 5, 5]]
    """
    vals = [list(v) for k,v in groupby(list_of_numbers)]
    idx = []
    a = range(len(list_of_numbers))
    i = 0
    for sublist in vals:
        j = i + len(sublist)
        idx.append(a[i:j])
        i = j
    return idx, vals


def permute(list_of_numbers, indices):
    return [list_of_numbers[i] for i in indices]

def group_by_indices(list_of_numbers, idx_groupings):
    return [[list_of_numbers[i] for i in g] for g in idx_groupings]

def invert_permutation(indices):
    return [i for i, j in sorted(enumerate(indices), key=lambda (_, j): j)]

def sort_group_perm(lengths):
    perm_idx, sorted_lengths = sort_decr(lengths)
    group_idx, group_lengths = group_by_element(sorted_lengths)
    inverse_perm_idx = invert_permutation(perm_idx)
    return perm_idx, group_idx, inverse_perm_idx

def sort_decr(lengths):
    perm_idx, sorted_lengths = zip(*[(c, d) for c, d in sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)])
    return perm_idx, sorted_lengths

def var_length_in_batch_wrapper(fn, inputs, inputs_xform, input_to_group_by, args):
    """
        1. permutes by length
        2. groups by length
        3. applies neural network
        4. unpermutes output of neural network

    args:
        fn: neural network function

    TODO:
        make sure that outputs.extend does not mess up the Variable
    """
    # sort by length
    lengths = [len(e) for e in input_to_group_by]
    perm_idx, sorted_lengths = sort_decr(lengths)
    inputs_p = map(lambda x: permute(x, perm_idx), inputs)
    # group by sorted length
    group_idx, group_lengths = group_by_element(sorted_lengths)
    inputs_grp = map(lambda x: group_by_indices(x, group_idx), inputs_p)
    # convert every group in inputs_grp to torch tensor 
    inputs_grp_th = map(lambda (f, y): map(f, y), zip(inputs_xform, inputs_grp))
    def execute_fn_on_grouped_inputs(fn, grouped_inputs):
        outputs = []
        for inp in zip(*grouped_inputs):
            out = fn(*inp)
            outputs.append(out)  # does not mess up the Variable.
        outputs = torch.cat(outputs)
        return outputs
    # run network
    outputs_p = execute_fn_on_grouped_inputs(fn, inputs_grp_th)  # Variable
    # unpermute
    inverse_perm_idx = invert_permutation(perm_idx)
    inverse_perm_idx_th = cuda_if_needed(torch.LongTensor(inverse_perm_idx), args)
    outputs = outputs_p[inverse_perm_idx_th]  # hopefully this doesn't mess up the gradient computation...
    return outputs

    # TODO: does this retain previous activations?
    # because you are running the network multiple times before the backward pass
    # yes, it still works because look at the "backward both losses together"
    #   https://discuss.pytorch.org/t/how-to-use-the-backward-functions-for-multiple-losses/1826/7?u=simonw

def var_length_var_dim_in_batch_wrapper(fn, inputs, inputs_xform, input_to_group_by, args):
    """
        1. permutes by length
        2. groups by length
        3. applies neural network
        4. unpermutes output of neural network

    args:
        fn: neural network function

    TODO:
        make sure that outputs.extend does not mess up the Variable
    """
    # sort by length
    lengths = [len(e) for e in input_to_group_by]
    perm_idx, sorted_lengths = sort_decr(lengths)
    inputs_p = map(lambda x: permute(x, perm_idx), inputs)
    # group by sorted length
    group_idx, group_lengths = group_by_element(sorted_lengths)
    inputs_grp = map(lambda x: group_by_indices(x, group_idx), inputs_p)
    # convert every group in inputs_grp to torch tensor 
    inputs_grp_th = map(lambda (f, y): map(f, y), zip(inputs_xform, inputs_grp))
    def execute_fn_on_grouped_inputs(fn, grouped_inputs):
        outputs = []
        for inp in zip(*grouped_inputs):
            out = fn(*inp)
            outputs.append(out)  # does not mess up the Variable.
        outputs = torch.cat(outputs)
        return outputs
    # run network
    outputs_p = execute_fn_on_grouped_inputs(fn, inputs_grp_th)  # Variable
    # unpermute
    inverse_perm_idx = invert_permutation(perm_idx)
    inverse_perm_idx_th = cuda_if_needed(torch.LongTensor(inverse_perm_idx), args)
    outputs = outputs_p[inverse_perm_idx_th]  # hopefully this doesn't mess up the gradient computation...
    return outputs


def reverse(x, dim):
    idx = torch.LongTensor([i for i in range(x.size(dim)-1, -1, -1)])
    if isinstance(x, torch.autograd.variable.Variable):
        idx = Variable(idx)
        if 'cuda' in x.data.type():
            idx = idx.cuda()
    else:
        assert 'Tensor' in x.type()
        if 'cuda' in x.type():
            idx = idx.cuda()
    return x.index_select(dim, idx)


def entropy(dist, eps=1e-20):
    dist_eps = dist + eps
    log_action_dist = torch.log(dist_eps)
    h = -torch.sum(log_action_dist * dist_eps)
    return h

def sample_from_categorical_dist(dist):
    m = Categorical(dist)
    s = m.sample()
    return s

def logprob_categorical_dist(dist, s):
    m = Categorical(dist)
    lp = m.log_prob(s)
    return lp







