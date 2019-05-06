import operator
import regex as re
from collections import OrderedDict
import datautils as du
import numpy as np

class Operator(object):
    def __init__(self):
        super(Operator, self).__init__()
        self.op_string = None
        self.matcher = None
        self.operator = None

    def transform(self, x, idx):
        raise NotImplementedError

class AttentionOperator(Operator):
    def __init__(self):
        super(AttentionOperator, self).__init__()
        # inherited: self.op_string, self.matcher, self.operator
        self.all_op_string = '[\+\-\*\/]'
        self.all_operator_matcher = re.compile(self.all_op_string)
        self.all_operators = OrderedDict([
            ('+', Plus),
            ('*', Multiply),
            ('-', Minus),
            ('/', Divide)
        ])
        self.static_operator_dict = OrderedDict([
            (operator.add, '+'),
            (operator.mul, '*'),
            (operator.sub, '-'),
            (operator.div, '/')
        ])

    def evaluate_subexp(self, x, idx=-1):
        terms, ops = du.extract_terms_ops(x, self.all_operator_matcher, self.all_operators)  # can be affected by np.inf
        if idx==-1 and self.operator in ops:
            idx = ops.index(self.operator)  # naive default
        if ops and idx < len(ops) and ops[idx] == self.operator:
            evaluated_subexp = self.operator(terms[idx], terms[idx+1])
            new_terms = terms[:idx] + [evaluated_subexp] + terms[idx+2:]
            new_ops = ops[:idx] + ops[idx+1:]
            return evaluated_subexp, new_terms, new_ops
        else:
            return None, None, None

    def get_new_exp_str(self, x, evaluated_subexp, new_terms, new_ops):
        if evaluated_subexp is not None:  # if evaluated_subexp actually does not work here
            new_exp_str = du.build_expression_string(new_terms, new_ops, self.static_operator_dict)
            return new_exp_str, evaluated_subexp
        else:
            return x, None

    def transform(self, x, idx=-1):
        evaluated_subexp, new_terms, new_ops = self.evaluate_subexp(x, idx)
        new_exp_str, evaluated_subexp = self.get_new_exp_str(x, evaluated_subexp, new_terms, new_ops)
        # evaluated_subexp is not actually modified
        return new_exp_str, evaluated_subexp

class Plus(AttentionOperator):
    def __init__(self):
        super(Plus, self).__init__()
        self.op_string = '+'
        self.operator = operator.add

class Minus(AttentionOperator):
    def __init__(self):
        super(Minus, self).__init__()
        self.op_string = '-'
        self.operator = operator.sub

class Multiply(AttentionOperator):
    def __init__(self):
        super(Multiply, self).__init__()
        self.op_string = '*'
        self.operator = operator.mul

class Divide(AttentionOperator):
    def __init__(self):
        super(Divide, self).__init__()
        self.op_string = '/'
        self.operator = operator.div