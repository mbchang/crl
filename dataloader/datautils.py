import numpy as np
from itertools import chain, combinations

def sample_term_in_range(interval):
    return np.random.randint(*interval)

def compute_prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def get_prime_factors(n):
    prime_factors = compute_prime_factors(np.abs(n))
    combos = chain.from_iterable(combinations(prime_factors, r) for r in range(len(prime_factors)+1))
    factors = list(set([int(np.prod(c)) for c in combos]))
    return factors

def extract_ops(new_exp_str, operator_matcher, operators):
    # find locations of operators and terms
    new_ops = []
    term_locations = []
    lo = 0
    for m in operator_matcher.finditer(new_exp_str):
        # op loc
        op_loc = m.span()
        # op
        new_op_str = new_exp_str[op_loc[0]:op_loc[1]]
        new_op = operators[new_op_str]().operator
        new_ops.append(new_op)
        # termloc
        hi = op_loc[0]
        term_loc = (lo, hi)
        term_locations.append(term_loc)
        lo = op_loc[1]
    term_locations.append((lo, len(new_exp_str)))
    return new_ops, term_locations

def extract_terms(new_exp_str, term_locations):
    # get terms
    new_terms = []
    for termloc in term_locations:
        new_term_str = new_exp_str[termloc[0]:termloc[1]]
        new_term = int(new_term_str)
        new_terms.append(new_term)
    return new_terms

def extract_terms_ops(new_exp_str, operator_matcher, operators):
    new_ops, term_locations = extract_ops(new_exp_str, operator_matcher, operators)
    new_terms = extract_terms(new_exp_str, term_locations)
    return new_terms, new_ops

def build_expression_string(terms, ops, operator_dict):
    exp_str = str(terms[0])
    for i in range(len(ops)):
        exp_str += operator_dict[ops[i]] + str(terms[i+1])
    return exp_str

def num2onehot(num, size):
    v = np.zeros(size)
    v[num] = 1
    return v