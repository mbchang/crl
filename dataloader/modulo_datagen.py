import numpy as np
import re
import operator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pprint
import copy

from collections import OrderedDict

from arithmetic import Plus, Minus, Multiply
import utils
import datautils as du

from datagen import ArithmeticDataGenerator

np.random.seed(0)

class ModuloDataGenerator(ArithmeticDataGenerator):
    def __init__(self, ops, numrange):
        super(ModuloDataGenerator, self).__init__(ops, numrange)
        self.range_length = max(self.range)
        self.modulus = self.range_length
        self.encoding_length = self.range_length + self.op_length

    def sample_next_term(self, op, first):
        if self.operator_dict[op] in '+-*':
            next_term = du.sample_term_in_range(self.range)
        else:
            assert False
        return next_term

    def create_problem(self, max_terms):
        num_ops = max_terms - 1
        ops = [self._sample_operator() for i in xrange(num_ops)]
        terms = [du.sample_term_in_range(self.range)]
        for i in xrange(num_ops):
            terms.append(self.sample_next_term(ops[i], terms[i-1]))
        exp_val = self.evaluate_expression(terms, ops)
        exp_str = du.build_expression_string(terms, ops, self.operator_dict)
        if self.verbose: print 'Final Expression: {} = {}'.format(exp_str, exp_val)
        return exp_str, exp_val, terms, ops

    def evaluate_expression(self, terms, ops):
        terms = copy.deepcopy(terms)
        ops = copy.deepcopy(ops)
        # first find all the ops that are * and /
        multiplicative_ops_indices = filter(lambda x: self.operator_dict[ops[x]] in '*/', range(len(ops)))

        while len(multiplicative_ops_indices) > 0:
            # evaluate multiplication
            multiplicative_op_index = multiplicative_ops_indices[0]  # get first multiplicative index
            surrounding_terms = terms[multiplicative_op_index], terms[multiplicative_op_index+1]
            multiplicative_val = ops[multiplicative_op_index](surrounding_terms[0], surrounding_terms[1]) % self.modulus  # modulo arithmetic
            # update ops and terms
            ops = ops[:multiplicative_op_index] + ops[multiplicative_op_index+1:]
            terms = terms[:multiplicative_op_index] + [multiplicative_val] + terms[multiplicative_op_index+2:]
            # check if there are more multiplicative terms
            multiplicative_ops_indices = filter(lambda x: self.operator_dict[ops[x]] in '*/', range(len(ops)))

        # at this point there are no multiplicative ops
        result = self._fold_left_ops_terms_eval(ops, terms)[-1] % self.modulus
        return result

def test_generate_data():
    dg = ModuloDataGenerator('+-*', [0,100])
    for i in range(1000):
        print '\n*******'
        exp_str, exp_val, all_terms, all_ops =  dg.create_problem(3)
        assert exp_val == dg.evaluate_expression(all_terms, all_ops)
        print '{}={}'.format(exp_str, exp_val)

if __name__ == '__main__':
    test_generate_data()

