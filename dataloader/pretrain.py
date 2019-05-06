import torch
import numpy as np

from multilingual_dataset import ArithmeticLanguageWordEncoding, ArithmeticLanguageTranslation

class PreTrainDataset(object):
    def __init__(self, max_terms, num_range, ops, samplefrom, episodecap, root, nlang):
        super(PreTrainDataset, self).__init__()

        envbuilder = lambda m, c, p: ArithmeticLanguageTranslation(
            max_terms=m, 
            num_range=num_range, 
            ops=ops, 
            samplefrom=samplefrom,
            episodecap=episodecap,
            cheat=c,
            root=root,
            curr=True,
            pair=p,
            nlang=nlang)

        self.pairs = self.create_pairs(nlang, 'ed')
        self.datasets = self.create_datasets(envbuilder, self.pairs, nlang)

        assert all(self.datasets[0].vocabulary == d.vocabulary for d in self.datasets)
        assert all(self.datasets[0].langsize == d.langsize for d in self.datasets)
        assert all(self.datasets[0].zsize == d.zsize for d in self.datasets)

        self.langsize = self.datasets[0].langsize
        self.zsize = self.datasets[0].zsize
        self.vocabulary = self.datasets[0].vocabulary
        self.vocabsize = self.datasets[0].vocabsize
        self.current_exp_strs = []

    def initialize_data(self, splits):
        map(lambda x: x.initialize_data(splits), self.datasets)

    def initialize_printer(self, logger, args):
        map(lambda x: x.initialize_printer(logger, args), self.datasets)

    def create_datasets(self, envbuilder, pairs, nlang):
        datasets = []
        # add translator dataset encoder-decoder
        print('ADDING TRANSLATOR DATASETS')
        for pair in pairs:
            print('PAIR', pair)
            datasets.append(envbuilder(m=[2,2,2], c=False, p=pair))
        # add reducer dataset
        print('ADDING REDUCER DATASETS')
        datasets.extend([
            envbuilder(m=[3,3,3], c=True, p='mm')])
        return datasets

    def create_pairs(self, nlang, mode):
        if mode == 'ed':
            pairs = ['em', 'me', 'pm', 'mp', 'rm', 'mr']
            if nlang >= 5:
                pairs += ['sm', 'ms']
            if nlang >= 6:
                pairs += ['gm', 'mg']
            if nlang >= 7:
                pairs += ['vm', 'mv']
            if nlang > 7: assert False
        elif mode == 'ring':
            # random permutation for ring: 4,1,3,2,5
            # r,m,p,e,s
            # m,p,e,s,r
            if nlang == 5:
                pairs = ['rm', 'mp', 'pe', 'es', 'sr']
            else:
                assert False
            pass
        else:
            assert False
        return pairs

    def change_dataset(self):
        self.d_index = np.random.randint(len(self.datasets))

    def reset(self, mode, z):
        return self.datasets[self.d_index].reset(mode, z, whole_expr=1)

    def encode_tokens(self, tokens):
        return self.datasets[self.d_index].encode_tokens(tokens)

    def decode_tokens(self, tokens):
        return self.datasets[self.d_index].decode_tokens(tokens)

    def get_exp_str(self, encoded_expression):
        return self.datasets[self.d_index].get_exp_str(encoded_expression)

    def add_exp_str(self, exp_str):
        self.datasets[self.d_index].add_exp_str(exp_str)

    def get_problem_trace(self):
        return self.datasets[self.d_index].get_problem_trace()

    def get_trace(self):
        return self.datasets[self.d_index].get_trace()

    def change_mt(self):
        self.datasets[self.d_index].change_mt()

class Pretrain_Multilingual_Dataset(object):
    def __init__(self, max_terms, num_range, ops, samplefrom, episodecap, root, nlang):
        super(Pretrain_Multilingual_Dataset, self).__init__()
        assert nlang == 5
        # just need to make sure these specs are the same as what was pretrained
        self.pretrain_env  = PreTrainDataset(
                max_terms=max_terms, 
                num_range=num_range, 
                ops=ops, 
                samplefrom=samplefrom,
                episodecap=episodecap,
                root=root,
                curr=True,
                nlang=nlang)

        self.multilingual_env = ArithmeticLanguageWordEncoding(
            max_terms=max_terms, 
            num_range=num_range, 
            ops=ops, 
            samplefrom=samplefrom,
            episodecap=episodecap,
            root=root, 
            curr=True,
            nlang=nlang
            )

        self.datasets = [self.pretrain_env, self.multilingual_env]

        assert all(self.datasets[0].vocabulary == d.vocabulary for d in self.datasets)
        assert all(self.datasets[0].langsize == d.langsize for d in self.datasets)
        assert all(self.datasets[0].zsize == d.zsize for d in self.datasets)

        self.langsize = self.datasets[0].langsize
        self.zsize = self.datasets[0].zsize
        self.vocabulary = self.datasets[0].vocabulary
        self.vocabsize = self.datasets[0].vocabsize
        self.current_exp_strs = []

    def initialize_data(self, splits):
        map(lambda x: x.initialize_data(splits), self.datasets)

    def initialize_printer(self, logger, args):
        map(lambda x: x.initialize_printer(logger, args), self.datasets)

    def change_dataset(self):
        self.d_index = np.random.randint(len(self.datasets))
        if self.d_index == 0:
            self.pretrain_env.change_dataset()

    def reset(self, mode, z):
        return self.datasets[self.d_index].reset(mode, z)

    def encode_tokens(self, tokens):
        return self.datasets[self.d_index].encode_tokens(tokens)

    def decode_tokens(self, tokens):
        return self.datasets[self.d_index].decode_tokens(tokens)

    def get_exp_str(self, encoded_expression):
        return self.datasets[self.d_index].get_exp_str(encoded_expression)

    def add_exp_str(self, exp_str):
        self.datasets[self.d_index].add_exp_str(exp_str)

    def get_problem_trace(self):
        return self.datasets[self.d_index].get_problem_trace()

    def get_trace(self):
        return self.datasets[self.d_index].get_trace()

    def change_mt(self):
        self.datasets[self.d_index].change_mt()

    def update_curriculum(self):
        self.datasets[1].update_curriculum()

