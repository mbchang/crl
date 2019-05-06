import torch
import numpy as np
import copy
import re
import itertools

from languages import Math_to_English, Math_to_Spanish, English_to_PigLatin, Spanish_to_PigSpanish, English_to_ReverseEnglish, Spanish_to_ReverseSpanish
from numerical_dataset import BaseArithmetic

class ArithmeticLanguage(BaseArithmetic):
    def __init__(self, max_terms, num_range, ops, samplefrom, episodecap, root, curr, nlang):
        super(ArithmeticLanguage, self).__init__(max_terms, num_range, ops, samplefrom, episodecap, root, curr)
        # translate or answer? 0 if translate, 1 if answer
        self.nlang = nlang
        assert self.nlang >= 4

        self.initialize_languages()
        self.language_pairings = list(itertools.product(self.languages, repeat=2))
        self.langsize = len(self.languages)
        self.zsize = 2
        self.initialize_pairings()

    def initialize_languages(self):
        self.languages = ['math', 'english', 'piglatin', 'reverseenglish'] # the ordering matters!!
        self.translators = {'math_to_english': Math_to_English(), 
                            'english_to_piglatin': English_to_PigLatin(), 
                            'english_to_reverseenglish': English_to_ReverseEnglish()}
        if self.nlang >= 5:
            self.languages.append('spanish')
            self.translators['math_to_spanish'] = Math_to_Spanish()
        if self.nlang >= 6:
            self.languages.append('pigspanish')
            self.translators['spanish_to_pigspanish'] = Spanish_to_PigSpanish()
        if self.nlang == 7:
            self.languages.append('reversespanish')
            self.translators['spanish_to_reversespanish'] = Spanish_to_ReverseSpanish()
        if self.nlang > 7: assert False

    def initialize_pairings(self):
        # random permutation: 4,1,3,2,5
        self.pairings = {}

        if self.nlang == 4:
            self.pairings['val'] = [('math', 'english'), ('piglatin', 'reverseenglish')]
            self.pairings['test'] = [('english', 'piglatin'), ('reverseenglish', 'math')]
            self.pairings['extrapval'] = [('english', 'piglatin'), ('reverseenglish', 'math')]
        elif self.nlang == 5:
            self.pairings['val'] = [('spanish', 'math'), ('math', 'english'), ('english', 'reverseenglish')]
            self.pairings['test'] = [('reverseenglish', 'piglatin'), ('piglatin', 'spanish')]
            self.pairings['extrapval'] = [('reverseenglish', 'piglatin'), ('piglatin', 'spanish')]
        elif self.nlang == 6:
            self.pairings['val'] = [('spanish', 'pigspanish'), ('pigspanish', 'math'), ('math', 'english')]
            self.pairings['test'] = [('english', 'reverseenglish'), ('reverseenglish', 'piglatin'), ('piglatin', 'spanish')]
            self.pairings['extrapval'] = [('english', 'reverseenglish'), ('reverseenglish', 'piglatin'), ('piglatin', 'spanish')]
        elif self.nlang == 7:
            self.pairings['val'] = [('spanish', 'pigspanish'), ('pigspanish', 'reversespanish'), ('reversespanish', 'math'), ('math', 'english')]
            self.pairings['test'] = [('english', 'reverseenglish'), ('reverseenglish', 'piglatin'), ('piglatin', 'spanish')]
            self.pairings['extrapval'] = [('english', 'reverseenglish'), ('reverseenglish', 'piglatin'), ('piglatin', 'spanish')]

        self.pairings['train'] = [p for p in self.language_pairings if p not in self.pairings['val'] + self.pairings['test']]

    def generate_all_possible_translations(self, numerical_p, numerical_a):
        english_p, english_a = map(self.translators['math_to_english'].translate,
            (numerical_p, numerical_a))
        pig_p, pig_a = map(self.translators['english_to_piglatin'].translate,
            (english_p, english_a))
        reverseenglish_p, reverseenglish_a = map(self.translators['english_to_reverseenglish'].translate,
            (english_p, english_a))

        possible_problems = {
            'math': (numerical_p, numerical_a),
            'english': (english_p, english_a),
            'piglatin': (pig_p, pig_a),
            'reverseenglish': (reverseenglish_p, reverseenglish_a)
        }

        if self.nlang >= 5:
            spanish_p, spanish_a = map(self.translators['math_to_spanish'].translate,
                (numerical_p, numerical_a))
            possible_problems['spanish'] = (spanish_p, spanish_a)
        if self.nlang >= 6:
            pigspanish_p, pigspanish_a = map(self.translators['spanish_to_pigspanish'].translate,
                (spanish_p, spanish_a))
            possible_problems['pigspanish'] = (pigspanish_p, pigspanish_a)
        if self.nlang == 7:
            reversespanish_p, reversespanish_a = map(self.translators['spanish_to_reversespanish'].translate,
                (spanish_p, spanish_a))
            possible_problems['reversespanish'] = (reversespanish_p, reversespanish_a)
        if self.nlang > 7: assert False

        return possible_problems

    def reset(self, mode, z):
        exp_str, exp_val, terms, ops = super(ArithmeticLanguage, self).reset(mode)

        # convert to numberical
        numerical_p, numerical_a = list(exp_str), list(str(exp_val))
        possible_problems = self.generate_all_possible_translations(numerical_p, numerical_a)

        # now sample languages
        pairing = self.pairings[mode][np.random.randint(len(self.pairings[mode]))]
        source_language_id, target_language_id = map(self.languages.index, pairing)

        # create input output pair
        inp = possible_problems[self.languages[source_language_id]][0]
        if z == 0:
            out = possible_problems[self.languages[target_language_id]][0]
        elif z == 1:
            out = possible_problems[self.languages[target_language_id]][1]
        else:
            raise ValueError

        trace = {"input": inp, "output": out, 
                 "source": self.languages[source_language_id], 
                 "target": self.languages[target_language_id], 
                 "z": 'answer' if z else 'translate'}
        self.trace = trace

        return inp, out, source_language_id, target_language_id, z  # (list, list, int, int, int)

    def get_trace(self):
        return self.trace

class ArithmeticLanguageEncoding(ArithmeticLanguage):
    def __init__(self, max_terms, num_range, ops, samplefrom, episodecap, root, curr, nlang):
        super(ArithmeticLanguageEncoding, self).__init__(max_terms, num_range, ops, samplefrom, episodecap, root, curr, nlang)
        self.vocabulary = NotImplementedError

    def reset(self, mode, z):
        inp, out, source_language_id, target_language_id, z = super(ArithmeticLanguageEncoding, self).reset(mode, z)

        # encode inp
        encoded_inp = self.encode_tokens(inp)

        # encode out
        encoded_out = self.encode_tokens(out)

        # assume we will have an embedding layer
        initial = (encoded_inp, target_language_id, z)
        target = encoded_out

        return initial, target

class ArithmeticLanguageWordEncoding(ArithmeticLanguageEncoding):
    def __init__(self, max_terms, num_range, ops, samplefrom, episodecap, root, curr, nlang):
        super(ArithmeticLanguageWordEncoding, self).__init__(max_terms, num_range, ops, samplefrom, episodecap, root, curr, nlang)
        # create vocabulary
        self.vocabulary = self.translators['math_to_english'].vocabulary.keys()
        for key, translator in self.translators.iteritems():
            self.vocabulary.extend(translator.vocabulary.values())
        self.vocabulary.append('STAHP')
        self.vocabsize = len(self.vocabulary)

        self.current_exp_strs = []

    def reset(self, mode, z):
        self.current_exp_strs = []
        return super(ArithmeticLanguageWordEncoding, self).reset(mode, z)

    def encode_tokens(self, tokens):
        """ assume input is a list """
        return map(self.vocabulary.index, tokens)

    def decode_tokens(self, tokens):
        answer = ' '.join([self.vocabulary[t] for t in tokens])
        return answer

    def get_exp_str(self, encoded_expression):
        # need to turn this into a string
        tokens = list(torch.max(encoded_expression, dim=-1)[1])
        decoded_tokens = self.decode_tokens(tokens)
        return decoded_tokens

    def add_exp_str(self, exp_str):
        self.current_exp_strs.append(exp_str)

    def get_problem_trace(self):
        return super(ArithmeticLanguageWordEncoding, self).get_trace()

    def get_trace(self):
        return str(copy.deepcopy(self.current_exp_strs)) + '\n\t{}'.format(self.get_problem_trace())

class ArithmeticLanguageTranslation(ArithmeticLanguageWordEncoding):
    def __init__(self, max_terms, num_range, ops, samplefrom, episodecap, root, curr, pair, nlang):
        super(ArithmeticLanguageTranslation, self).__init__(max_terms, num_range, ops, samplefrom, episodecap, root, curr, nlang)

        # code
        self.code = {
            'm': 'math', 
            'e': 'english', 
            'r': 'reverseenglish', 
            'p': 'piglatin',
            's': 'spanish',
            'g': 'pigspanish',
            'v': 'reversespanish'}

        self.pair = (self.code[pair[0]], self.code[pair[1]])

        # here are the valid pairings
        # train, val, test are all the same pairings here
        self.generate_pairings()

        assert self.pair in self.enc_dec_pairings or self.pair in self.ring_pairings

        self.pairings = {k: [self.pair] for k in ['train', 'val', 'test']}

        self.current_exp_strs = []

    def generate_pairings(self):
        # here are the valid pairings
        # train, val, test are all the same pairings here
        self.enc_dec_pairings = {
            # in which case z is always 1
            ('math', 'math'), 
            # in which case z is always 0 
            ('english', 'math'), ('math', 'english'),
            ('piglatin', 'math'), ('math', 'piglatin'),
            ('reverseenglish', 'math'), ('math', 'reverseenglish'),
        }
        if self.nlang >= 5:
            self.enc_dec_pairings.update({('spanish', 'math'), ('math', 'spanish')})
             # ['rm', 'mp', 'pe', 'es', 'sr']
            self.ring_pairings = {
                ('reverseenglish', 'math'), ('math', 'piglatin'),
                ('piglatin', 'english'), ('english', 'spanish'),
                ('spanish', 'reverseenglish')
            }

        if self.nlang >= 6:
            self.enc_dec_pairings.update({('pigspanish', 'math'), ('math', 'pigspanish')})
        if self.nlang == 7:
            self.enc_dec_pairings.update({('reversespanish', 'math'), ('math', 'reversespanish')})
        if self.nlang > 7: assert False
        
    def reset(self, mode, z, whole_expr=None):
        self.current_exp_strs = []
        if self.pair == ('math', 'math'):
            z = 1
            problem = super(ArithmeticLanguageTranslation, self).reset(mode, z)
        else:
            z = 0
            problem = super(ArithmeticLanguageTranslation, self).reset(mode, z)
            assert whole_expr is not None
            # 1 if we want whole expr, 0 if just want answer
            if whole_expr == 1:
                pass  # keep as is
            elif whole_expr == 0:
                initial, target = problem
                p, t, z = initial
                initial = ([p[0]], t, z)
                target = [target[0]]
                problem = initial, target
            else:
                assert False
        self.add_exp_str(self.decode_tokens(problem[0][0]))
        return problem

    def get_exp_str(self, encoded_expression):
        # need to turn this into a string
        tokens = list(torch.max(encoded_expression, dim=-1)[1])
        decoded_tokens = self.decode_tokens(tokens)
        return decoded_tokens

    def add_exp_str(self, exp_str):
        self.current_exp_strs.append(exp_str)

    def get_problem_trace(self):
        return super(ArithmeticLanguageTranslation, self).get_problem_trace()

    def get_trace(self):
        return copy.deepcopy(self.current_exp_strs)

