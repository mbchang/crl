import numpy as np
import re
import operator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pprint

from collections import OrderedDict

from arithmetic import Plus, Minus, Multiply, Divide
import utils
import datautils as du

np.random.seed(0)

class ArithmeticDataGenerator(object):
    def __init__(self, ops, numrange):
        super(ArithmeticDataGenerator, self).__init__()
        """
            should set self.encoding_length
        """
        self.verbose = False

        # Mappings
        self.static_operator_dict = OrderedDict([
                (operator.add, '+'),
                (operator.mul, '*'),
                (operator.sub, '-'),
                (operator.div, '/')
            ])
        self.reverse_static_operator_dict = {v:k for (k, v) in self.static_operator_dict.items()}
        self.operator_dict = OrderedDict()  # this needs to be orderedict because of onehot
        for op in ops:
            if op in self.reverse_static_operator_dict:
                self.operator_dict[self.reverse_static_operator_dict[op]] = op
        self.operator_names = self.operator_dict.keys()
        self.operators = self.operator_dict.values()

        # Parameters
        self.range = numrange
        assert self.range[0] == min(self.range) == 0

        # Stats
        self.op_length = len(self.operator_dict)
        self.range_length = None
        self.encoding_length = None

    ######################### Internal Methods #################################

    def _sample_operator(self):
        return np.random.choice(self.operator_names)

    def _fold_left_ops_terms_sample(self, ops, terms):
        result_so_far = terms[0]
        for o in ops:
            next_term = self._sample_second_term(o, result_so_far)
            result_so_far = o(result_so_far, next_term)
            terms.append(next_term)
        return ops, terms, result_so_far

    def _fold_left_ops_terms_eval(self, ops, terms):
        results_so_far = [terms[0]]
        for i in range(len(ops)):
            o = ops[i]
            next_term = terms[i+1]
            result_so_far = o(results_so_far[-1], next_term)
            results_so_far.append(result_so_far)
        return results_so_far

    def _extract_multiplicative_groups(self, ops):
        # find the indices of the ops
        multiplicative_ops_indices = filter(lambda x: self.operator_dict[ops[x]] in '*/', range(len(ops)))
        multiplicative_groups = utils.group_consecutive(multiplicative_ops_indices)
        return multiplicative_groups

    def _combine_ops_terms(self, ops, terms):
        expression = [terms[0]]
        for i in range(len(ops)):
            expression.append(ops[i])
            expression.append(terms[i+1])
        return expression

    def _split_expression(self, expression):
        ops = expression[1::2]
        terms = expression[::2]
        return ops, terms

    ###################### Helper External Methods #############################

    def get_additive_expression(self, ops, additive_ops_indices):
        additive_ops = [ops[i] for i in additive_ops_indices]
        additive_terms = [du.sample_term_in_range(self.range)]
        additive_ops, additive_terms, exp_val = self._fold_left_ops_terms_sample(additive_ops, additive_terms)
        if self.verbose:
            print 'Additive expression {} = {}'.format(
                du.build_expression_string(additive_terms, additive_ops, self.operator_dict),
                exp_val)
        return additive_ops, additive_terms, exp_val

    def create_multiplicative_expression(self):
        raise NotImplementedError

    def match_multiplicative_groups_to_additive_term(self, multiplicative_groups):
        multiplicative_groups_to_additive_term = {}
        for i, mg in enumerate(multiplicative_groups):
            lo_index = mg[0]
            if lo_index > 0:
                # we are matching terms that are not the first term
                # in this case, the term we match corresponds the
                # index of the additive term that corresponds to this
                # multiplicative group
                if multiplicative_groups[0][0] == 0:
                    additive_term_to_match_index = i
                else:
                    additive_term_to_match_index = i+1
            else:
                # we are matching the first term
                additive_term_to_match_index = 0
            multiplicative_groups_to_additive_term[mg] = additive_term_to_match_index
        return multiplicative_groups_to_additive_term

    def interleave_additive_multiplicative(self, additive_terms, additive_ops,
                                    additive_term_to_multiplicative_group):
        all_terms = []
        all_ops = []
        for j in xrange(len(additive_terms)):
            if j in additive_term_to_multiplicative_group:
                all_terms.extend(additive_term_to_multiplicative_group[j][1])
                all_ops.extend(additive_term_to_multiplicative_group[j][0])
            else:
                all_terms.append(additive_terms[j])
            if len(additive_ops) > 0 and j < len(additive_terms)-1:
                all_ops.append(additive_ops[j])
        return all_ops, all_terms

    ######################### External Methods #################################

    def evaluate_expression(self, terms, ops):
        assert len(ops) == len(terms)-1
        expression = self._combine_ops_terms(ops, terms)
        while self._extract_multiplicative_groups(ops) != []:
            mg = self._extract_multiplicative_groups(ops)[0]
            mg = tuple([1 + 2*i for i in mg])
            lo, hi = mg[0], mg[-1]
            multiplicative_ops = [expression[i] for i in mg]
            multiplicative_terms = [expression[i] for i in range(lo-1, hi+2) if i not in mg]  # check this
            multiplicative_val = self._fold_left_ops_terms_eval(multiplicative_ops, multiplicative_terms)[-1]
            # replace everything from [lo-1, hi+2) with multiplicative_val
            expression = expression[:lo-1] + [multiplicative_val] + expression[hi+2:]
            ops, terms = self._split_expression(expression)
        # at this point, it should just be an expression of additive terms
        additive_ops, additive_terms = self._split_expression(expression)
        assert all(self.operator_dict[ao] in '+-' for ao in additive_ops)
        result = self._fold_left_ops_terms_eval(additive_ops, additive_terms)[-1]
        return result

    def create_problem(self, max_terms):
        """
            all the input, intermediate, and output terms should be
                in self.range, inclusive

            expressions contain up to self.max_terms-1 operators

            divisions are always divisible, never divisible by 0

            only positive integers
        """
        num_ops = max_terms - 1
        ops = [self._sample_operator() for i in xrange(num_ops)]

        # focus on the additive_ops_first
        additive_ops_indices = filter(lambda x: self.operator_dict[ops[x]] in '+-', range(len(ops)))
        additive_ops, additive_terms, exp_val = self.get_additive_expression(ops, additive_ops_indices)

        # match a multiplicative expression for each additive term
        multiplicative_groups = self._extract_multiplicative_groups(ops)
        additive_term_to_multiplicative_group = self.create_multiplicative_expression(
            ops, multiplicative_groups, additive_terms)

        # put everything together
        all_ops, all_terms = self.interleave_additive_multiplicative(additive_terms, additive_ops, additive_term_to_multiplicative_group)
        exp_str = du.build_expression_string(all_terms, all_ops, self.operator_dict)
        if self.verbose: print 'Final Expression: {} = {}'.format(exp_str, exp_val)
        return exp_str, exp_val, all_terms, all_ops

    def encode_problem(self, exp_str, exp_val, terms, ops):
        raise NotImplementedError