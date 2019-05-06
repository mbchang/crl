import copy
from collections import namedtuple
from enum import Enum
import itertools
import numpy as np
import pprint
import torch

from image_transforms import Rotate, Scale, Translate, Identity, CurriculumRotate, CurriculumScale, CurriculumTranslate, CurriculumIdentity

xform = namedtuple('Transformation', ('function', 'name', 'id'))

def find_num_decimal_places(x):
    assert isinstance(x, float)
    if x % 1 == 0:
        return 0
    else:
        return 1 + find_num_decimal_places(x*10)

def linear_curr(lo_limit, hi_limit, incr):
    curr = np.arange(start=lo_limit, stop=hi_limit, step=incr)
    assert max(curr) == curr[-1]
    if max(curr) < hi_limit:
        curr = np.concatenate((curr, np.array([hi_limit])))
    decimals = 2  # HARDCODED
    curr = np.around(curr, decimals=decimals)
    return curr

class TransformationConfig(object):
    def __init__(self, cuda=False):
        super(TransformationConfig, self).__init__()
        self.cuda = cuda
        self.define_individual_transformations_linear_curriculum()
        self.all_transformations = self.define_group_transformations()
        self.transformation_combinations = {}

    def __eq__(self, other):
        return self.all_transformations == other.all_transformations

    def get_all_transformations(self):
        return [t.function for t in self.all_transformations]

    def get_transformation_combination(self, dataset, mode):
        return self.transformation_combinations[dataset][mode] 

    def get_dataset_transformations(self, dataset):
        return self.transformation_combinations[dataset]['transformations']

    def define_individual_transformations(self):
        raise NotImplementedError

    def define_group_transformations(self):
        raise NotImplementedError

    def update_curriculum(self):
        for t in self.all_transformations:
            t.function.update_curriculum()

class SpatialImageTransformations(TransformationConfig):
    def __init__(self, rp=60, sp=0.6, tsp=0.38, tnp=0.29, tbp=0.2, cuda=False):
        """
        This entire method should be run in this particular order.
        The purpose of grouping the operations together is not to encourage
        interchangeability of the ordering but rather to just make it easier to look at

        self.transformation_combinations[name][mode] = [tuples of indices into self.all_transformations]

        The default parameters are calibrated to 64x64 images

        self.rotate_param = 60
        self.scale_param = 0.6
        self.translate_small_param = 0.38
        self.translate_normal_param = 0.29
        self.translate_big_param = 0.2
        """
        self.rotate_param = rp
        self.scale_param = sp
        self.translate_small_param = tsp
        self.translate_normal_param = tnp
        self.translate_big_param = tbp

        super(SpatialImageTransformations, self).__init__(cuda)

        self.define_3c1()
        self.define_3c2_RT()
        self.define_3c3_SRT()
        self.define_identity()
        self.define_3c2_RT_full()
        self.define_translate_all_normal()
        self.define_translate_left_normal()

    def define_individual_transformations(self):
        self.rotate_left = xform(Rotate(-self.rotate_param, cuda=self.cuda), 'rotate_left', 0)
        self.rotate_right = xform(Rotate(self.rotate_param, cuda=self.cuda), 'rotate_right', 1)

        self.scale_small = xform(Scale(np.round(1.0/self.scale_param, decimals=1), cuda=self.cuda), 'scale_small', 2)
        self.scale_big = xform(Scale(self.scale_param, cuda=self.cuda), 'scale_big', 3)

        self.translate_up_small = xform(Translate(self.translate_small_param, 0, cuda=self.cuda), 'translate_up_small', 4)
        self.translate_down_small = xform(Translate(-self.translate_small_param, 0, cuda=self.cuda), 'translate_down_small', 5)
        self.translate_left_small = xform(Translate(0, self.translate_small_param, cuda=self.cuda), 'translate_left_small', 6)
        self.translate_right_small = xform(Translate(0, -self.translate_small_param, cuda=self.cuda), 'translate_right_small', 7)

        self.translate_up_normal = xform(Translate(self.translate_normal_param, 0, cuda=self.cuda), 'translate_up_normal', 8)
        self.translate_down_normal = xform(Translate(-self.translate_normal_param, 0, cuda=self.cuda), 'translate_down_normal', 9)
        self.translate_left_normal = xform(Translate(0, self.translate_normal_param, cuda=self.cuda), 'translate_left_normal', 10)
        self.translate_right_normal = xform(Translate(0, -self.translate_normal_param, cuda=self.cuda), 'translate_right_normal', 11)

        self.translate_up_big = xform(Translate(self.translate_big_param, 0, cuda=self.cuda), 'translate_up_big', 12)
        self.translate_down_big = xform(Translate(-self.translate_big_param, 0, cuda=self.cuda), 'translate_down_big', 13)
        self.translate_left_big = xform(Translate(0, self.translate_big_param, cuda=self.cuda), 'translate_left_big', 14)
        self.translate_right_big = xform(Translate(0, -self.translate_big_param, cuda=self.cuda), 'translate_right_big', 15)

        self.identity = xform(Identity(cuda=self.cuda), 'identity', 16)

    def define_individual_transformations_linear_curriculum(self):
        self.rotate_left = xform(CurriculumRotate([-self.rotate_param], cuda=self.cuda), 'rotate_left', 0)
        self.rotate_right = xform(CurriculumRotate([self.rotate_param], cuda=self.cuda), 'rotate_right', 1)

        self.scale_small = xform(CurriculumScale([np.round(1.0/self.scale_param, decimals=1)], cuda=self.cuda), 'scale_small', 2)
        self.scale_big = xform(CurriculumScale([self.scale_param], cuda=self.cuda), 'scale_big', 3)

        # this is a linear schedule
        lo = 0
        incr = 0.01
        up_curr_gen = lambda hi: [(c, 0) for c in linear_curr(lo, hi, incr)]
        down_curr_gen = lambda hi: [(-c, 0) for c in linear_curr(lo, hi, incr)]
        left_curr_gen = lambda hi: [(0, c) for c in linear_curr(lo, hi, incr)]
        right_curr_gen = lambda hi: [(0, -c) for c in linear_curr(lo, hi, incr)]

        self.translate_up_small = xform(CurriculumTranslate(up_curr_gen(self.translate_small_param), cuda=self.cuda), 'translate_up_small', 4)
        self.translate_down_small = xform(CurriculumTranslate(down_curr_gen(self.translate_small_param), cuda=self.cuda), 'translate_down_small', 5)
        self.translate_left_small = xform(CurriculumTranslate(left_curr_gen(self.translate_small_param), cuda=self.cuda), 'translate_left_small', 6)
        self.translate_right_small = xform(CurriculumTranslate(right_curr_gen(self.translate_small_param), cuda=self.cuda), 'translate_right_small', 7)

        self.translate_up_normal = xform(CurriculumTranslate(up_curr_gen(self.translate_normal_param), cuda=self.cuda), 'translate_up_normal', 8)
        self.translate_down_normal = xform(CurriculumTranslate(down_curr_gen(self.translate_normal_param), cuda=self.cuda), 'translate_down_normal', 9)
        self.translate_left_normal = xform(CurriculumTranslate(left_curr_gen(self.translate_normal_param), cuda=self.cuda), 'translate_left_normal', 10)
        self.translate_right_normal = xform(CurriculumTranslate(right_curr_gen(self.translate_normal_param), cuda=self.cuda), 'translate_right_normal', 11)

        self.translate_up_big = xform(CurriculumTranslate(up_curr_gen(self.translate_big_param), cuda=self.cuda), 'translate_up_big', 12)
        self.translate_down_big = xform(CurriculumTranslate(down_curr_gen(self.translate_big_param), cuda=self.cuda), 'translate_down_big', 13)
        self.translate_left_big = xform(CurriculumTranslate(left_curr_gen(self.translate_big_param), cuda=self.cuda), 'translate_left_big', 14)
        self.translate_right_big = xform(CurriculumTranslate(right_curr_gen(self.translate_big_param), cuda=self.cuda), 'translate_right_big', 15)

        self.identity = xform(CurriculumIdentity(cuda=self.cuda), 'identity', 16)

    def define_group_transformations(self):
        self.rotate = [self.rotate_left, self.rotate_right]
        self.scale = [self.scale_small, self.scale_big]
        self.translate_small = [self.translate_up_small, self.translate_down_small, self.translate_left_small, self.translate_right_small]
        self.translate_normal = [self.translate_up_normal, self.translate_down_normal, self.translate_left_normal, self.translate_right_normal]
        self.translate_big = [self.translate_up_big, self.translate_down_big, self.translate_left_big, self.translate_right_big]
        self.translate = self.translate_small + self.translate_normal + self.translate_big
        all_transformations = self.rotate + self.scale + self.translate + [self.identity]

        # checks
        for i, t in enumerate(all_transformations):
            assert t.id == i
        return all_transformations

    def define_identity(self):
        self.transformation_combinations['identity'] = {'train': [], 'val': [], 'test': [],
            'transformations': [self.identity], 'depth': 1}
        for mode in ['train', 'val', 'test']:
            for t in self.transformation_combinations['identity']['transformations']:
                self.transformation_combinations['identity'][mode].append((t.id,))

    def define_translate_all_normal(self):
        self.transformation_combinations['T_all_normal'] = {'train': [], 'val': [], 'test': [],
            'transformations': self.translate_normal, 'depth': 1}
        for mode in ['train', 'val', 'test']:
            for t in self.transformation_combinations['T_all_normal']['transformations']:
                self.transformation_combinations['T_all_normal'][mode].append((t.id,))

    def define_translate_left_normal(self):
        self.transformation_combinations['T_left_normal'] = {'train': [], 'val': [], 'test': [],
            'transformations': [self.translate_left_normal], 'depth': 1}
        for mode in ['train', 'val', 'test']:
            for t in self.transformation_combinations['T_left_normal']['transformations']:
                self.transformation_combinations['T_left_normal'][mode].append((t.id,))

    def define_3c1(self):
        self.transformation_combinations['3c1'] = {'train': [], 'val': [], 'test': [],
            'transformations': self.rotate+self.scale+self.translate_normal, 'depth': 1}
        for mode in ['train', 'val', 'test']:
            for t in self.transformation_combinations['3c1']['transformations']:
                self.transformation_combinations['3c1'][mode].append((t.id,))

    def define_3c2_RT(self):
        self.transformation_combinations['3c2_RT'] = {'train': [], 'val': [], 'test': [], 'depth': 2}
        # held out
        val_held_out = [('scale_big', 'translate_left_big'), ('rotate_right', 'translate_up_normal')]
        test_held_out = [('scale_small', 'translate_right_small'), ('rotate_left', 'translate_down_normal')]
        # cartesian product
        scale_translate_small = list(itertools.product([self.scale_small], self.translate_small))  # (1 x 4)
        scale_translate_big = list(itertools.product([self.scale_big], self.translate_big))  # (1 x 4)
        rotate_translate = list(itertools.product(self.rotate, self.translate_normal))  # (2 x 4)
        scale_rotate = list(itertools.product(self.scale, self.rotate))  # (2 x 2)
        # all combinations for this dataset
        self.transformation_combinations['3c2_RT']['transformations'] = scale_translate_small + scale_translate_big + rotate_translate + scale_rotate
        # assign to different modes
        for t in self.transformation_combinations['3c2_RT']['transformations']:
            if (t[0].name, t[1].name) in val_held_out:
                self.transformation_combinations['3c2_RT']['val'].append((t[0].id, t[1].id))
            elif (t[0].name, t[1].name) in test_held_out:
                self.transformation_combinations['3c2_RT']['test'].append((t[0].id, t[1].id))
            else:
                self.transformation_combinations['3c2_RT']['train'].append((t[0].id, t[1].id))

    def define_3c2_RT_full(self):
        self.transformation_combinations['3c2_RT_full'] = {'train': [], 'val': [], 'test': [], 'depth': 2}
        # cartesian product
        scale_translate_small = list(itertools.product([self.scale_small], self.translate_small))  # (1 x 4)
        scale_translate_big = list(itertools.product([self.scale_big], self.translate_big))  # (1 x 4)
        rotate_translate = list(itertools.product(self.rotate, self.translate_normal))  # (2 x 4)
        scale_rotate = list(itertools.product(self.scale, self.rotate))  # (2 x 2)
        # all combinations for this dataset
        self.transformation_combinations['3c2_RT_full']['transformations'] = scale_translate_small + scale_translate_big + rotate_translate + scale_rotate
        # assign to different modes
        for mode in ['train', 'val', 'test']:
            for t in self.transformation_combinations['3c2_RT_full']['transformations']:
                self.transformation_combinations['3c2_RT_full'][mode].append((t[0].id, t[1].id))

    def define_3c3_SRT(self):
        self.transformation_combinations['3c3_SRT'] = {'train': [], 'val': [], 'test': [], 'depth': 3}
        scale_rotate_translate_small = list(itertools.product([self.scale_small], self.rotate, self.translate_small))  # (1 x 2 x 4)
        scale_rotate_translate_big = list(itertools.product([self.scale_big], self.rotate, self.translate_big))  # (1 x 2 x 4)
        # all combinations for this dataset
        self.transformation_combinations['3c3_SRT']['transformations'] = scale_rotate_translate_small + scale_rotate_translate_big
        # assign to different modes
        for mode in ['train', 'val', 'test']:
            for t in self.transformation_combinations['3c3_SRT']['transformations']:
                self.transformation_combinations['3c3_SRT'][mode].append((t[0].id, t[1].id, t[2].id))

    def define_3c3_STR(self):
        self.transformation_combinations['3c3_SRT'] = {'train': [], 'val': [], 'test': [], 'depth': 3}
        scale_translate_rotate_small = list(itertools.product([self.scale_small], self.translate_small, self.rotate))  # (1 x 4 x 2)
        scale_translate_rotate_big = list(itertools.product([self.scale_big], self.translate_big, self.rotate))  # (1 x 4 x 2)
        # all combinations for this dataset
        self.transformation_combinations['3c3_SRT']['transformations'] = scale_translate_rotate_small + scale_translate_rotate_big
        # assign to different modes
        for mode in ['train', 'val', 'test']:
            for t in self.transformation_combinations['3c3_SRT']['transformations']:
                self.transformation_combinations['3c3_SRT'][mode].append((t[0].id, t[1].id, t[2].id))

class ConcatTransformationCombiner(object):
    def __init__(self, transformation_combiners):
        self.transformation_combiners = transformation_combiners
        self.verify_consistency()

    def sample(self):
        idx = np.random.randint(len(self.transformation_combiners))
        transformation_combiner = self.transformation_combiners[idx]
        combo, combo_info = transformation_combiner.sample()
        return combo, combo_info

    def verify_consistency(self):
        """
            supposed to verify that the transformation_combiner
            object for each mode has the same set of transformations
        """
        retrieve_tc = lambda x: x.transform_config
        all_tcs = [retrieve_tc(x) for x in self.transformation_combiners]
        assert all_tcs.count(all_tcs[0]) == len(all_tcs)

    def get_transform_config(self):
        return self.transformation_combiners[0].transform_config  # assume all are equal

    def update_curriculum(self):
        for tc in self.transformation_combiners:
            tc.update_curriculum()

class BaseTransformationCombiner(object):
    def __init__(self, name='', transformations=None, transformation_combinations=None):
        super(BaseTransformationCombiner, self).__init__()
        """
            self.transformations: list of Transform objects
            self.transformation_combinations: list of tuples of indices into self.transformations
        """
        self.transformations = transformations
        self.transformation_combinations = transformation_combinations
        self.name = name     

    def sample(self):
        dataset_id = np.random.randint(len(self.transformation_combinations))
        xform_ids = self.transformation_combinations[dataset_id]
        dormant_transformations = [self.transformations[x] for x in xform_ids]
        combo = tuple([t() for t in dormant_transformations])
        combo_parameters = tuple([t.get_parameter() for t in dormant_transformations])
        inverse_combo = tuple([t.invert() for t in dormant_transformations])
        inverse_parameters = tuple([t.get_inverse_parameter() for t in dormant_transformations])

        combo_info = {
            'name': self.name,
            'ids': xform_ids,
            'forward': combo,
            'inverse': inverse_combo,
            'forward_parameters': combo_parameters,
            'inverse_parameters': inverse_parameters,
            'depth': len(xform_ids)
        }  # note that you are passing a dictionary, which is mutable

        return combo, combo_info

class TransformationCombiner(BaseTransformationCombiner):
    def __init__(self, transform_config, name='', mode='', cuda=False):
        assert name
        assert mode
        self.transform_config = transform_config
        super(TransformationCombiner, self).__init__(
            name=name,
            transformations=self.transform_config.get_all_transformations(), 
            transformation_combinations=self.transform_config.get_transformation_combination(name, mode))
        self.dataset_transformations = self.transform_config.get_dataset_transformations(name)

    def get_transform_config(self):
        return self.transform_config

    def update_curriculum(self):
        self.transform_config.update_curriculum()


