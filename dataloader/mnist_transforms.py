import torch
import copy
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pprint
from torchvision import datasets, transforms
import torchsample
from tqdm import tqdm

from base_dataloader import DataLoader
from image_transforms import *
from transformation_combination import TransformationCombiner
from generative_recognition_mapping import GR_Map_full 

torch.manual_seed(0)
np.random.seed(0)

def shrink_mnist_dataset(mnist_orig, bkgd_dim):
    mnist_shrunk = {}
    for k in mnist_orig.keys():
        v_data, v_labels = mnist_orig[k]
        mnist_shrunk[k] = (place_subimage_in_background(bkgd_dim)(v_data), 
            v_labels)
    return mnist_shrunk

def cuda_mnist_dataset(mnist_dataset):
    for key in mnist_dataset.keys():
        inputs = mnist_dataset[key][0]
        targets = mnist_dataset[key][1]
        mnist_dataset[key] = (inputs.cuda(), targets.cuda())
    return mnist_dataset

class BaseImageTransformDataLoader(DataLoader):
    def __init__(self, cuda):
        super(BaseImageTransformDataLoader, self).__init__()
        self.cuda = cuda

    def set_xform_combo_info(self, xform_combo_info):
        self.xform_combo_info = xform_combo_info

    def get_xform_combo_info(self):
        return copy.deepcopy(self.xform_combo_info)

    def initialize_data(self, splits):
        pass

    def reset(self, mode, bsize):
        return self.transformation_dataloader.reset(
            mode, bsize)

    def get_trace(self):
        return ''

    def change_mt(self):
        pass

class ImageTransformDataLoader_identity(BaseImageTransformDataLoader):
    def __init__(self, dataset, composition_depth, cuda=False):
        super(ImageTransformDataLoader_identity, self).__init__(cuda)
        self.transformations = [
            Identity(cuda=self.cuda)
        ]
        self.composition_depth = composition_depth
        self.transformation_dataloader = TransformationDataloader(
            dataset, self.transformations, self.composition_depth,
            self.set_xform_combo_info, splittype='all')
        self.num_train = self.transformation_dataloader.num_train
        self.num_test = self.transformation_dataloader.num_test
        self.xform_combo_info = None

    def get_composition_depth(self):
        return self.composition_depth

class TransformationCombinationDataLoader(BaseImageTransformDataLoader):
    def __init__(self, dataset, transformation_combinations, transform_config, num_transfer=1, cuda=False):
        super(TransformationCombinationDataLoader, self).__init__(cuda)
        self.dataset = dataset
        self.transformation_combinations = transformation_combinations
        self.transform_config = transform_config
        self.num_transfer = num_transfer
        self.dataloaders = {}
        for key in ['train', 'val', 'test']:
            len_dataset = len(dataset[key][0])
            num_transfer = self.num_transfer if key == 'train' else 1
            clipped_len_dataset = int(num_transfer*len_dataset)
            inputs = dataset[key][0][:clipped_len_dataset]
            targets = dataset[key][1][:clipped_len_dataset]

            self.dataloaders[key] = BasicTransformDataLoader(
                inputs=inputs,
                targets=targets,
                transformation_combination=transformation_combinations[key],
                set_xform_combo_info_callback=self.set_xform_combo_info)
        self.num_train = len(self.dataloaders['train'])
        self.num_test = len(self.dataloaders['test'])
        self.verify_consistency()
        self._gr_map = GR_Map_full(self.transform_config).get_gr_map()

    def reset(self, mode, bsize):
        inputs, targets = self.dataloaders[mode].next(bsize)
        assert inputs.size(2) == inputs.size(3) == 64
        return inputs, targets

    def get_composition_depth(self):
        xform_combo_info = self.get_xform_combo_info()
        return xform_combo_info['depth']

    def get_trace(self):
        xform_combo_info = self.get_xform_combo_info()
        forward_parameters = xform_combo_info['forward_parameters']
        inverse_parameters = xform_combo_info['inverse_parameters']
        trace = 'forward parameters: {} inverse parameters: {}'.format(forward_parameters, inverse_parameters)
        return trace

    def update_curriculum(self):
        for key in ['train', 'val', 'test']:
            self.dataloaders[key].transformation_combination.update_curriculum()

    def verify_consistency(self):
        """
            supposed to verify that the transformation_combiner
            object for each mode has the same set of transformations
        """
        retrieve_tc = lambda x: self.dataloaders[x].transformation_combination.get_transform_config()
        train_tc_at = retrieve_tc('train').all_transformations
        val_tc_at = retrieve_tc('val').all_transformations
        test_tc_at = retrieve_tc('test').all_transformations
        assert train_tc_at == val_tc_at == test_tc_at == self.transform_config.all_transformations

    def get_gr_map(self):
        return copy.deepcopy(self._gr_map)

    def decode_tokens(self, x):
        return ''.join([str(y) for y in x])

class BasicDataLoader(torch.utils.data.TensorDataset):
    def __init__(self, inputs, targets):
        super(BasicDataLoader, self).__init__(inputs, targets)
        self.inputs = inputs
        self.targets = targets
        self.counter = 0

    def size(self):
        return self.inputs.size()

    def permute(self):
        perm = torch.LongTensor(np.random.permutation(range(len(self.inputs))))
        if self.inputs.is_cuda:
            perm = perm.cuda()
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self, bsize):
        if self.counter >= len(self.inputs)-bsize:
            self.permute()
            self.counter = 0
        inputs = self.inputs[self.counter: self.counter+bsize]
        targets = self.targets[self.counter: self.counter+bsize]
        self.counter += bsize
        return inputs, targets

class BasicTransformDataLoader(BasicDataLoader):
    def __init__(self, inputs, targets, transformation_combination, set_xform_combo_info_callback):
        """     
            inputs: (bsize, 1, H, W)
            targets: (bsize, 1)
            transformation_combination: TransformationCombiner object
        """
        super(BasicTransformDataLoader, self).__init__(inputs, targets)
        self.transformation_combination = transformation_combination
        self.set_xform_combo_info_callback = set_xform_combo_info_callback

    def get_composition_depth(self):
        pass

    def set_xform_combo_info(self, xform_combo_info):
        self.set_xform_combo_info_callback(xform_combo_info)

    def next(self, bsize):
        # get the raw data
        inputs, targets = super(BasicTransformDataLoader, self).next(bsize)
        # transform it
        xform_combo, combo_info = self.transformation_combination.sample()
        self.set_xform_combo_info(combo_info)
        # compose the transformations
        inputs = transforms.Compose(xform_combo)(inputs)
        return inputs, targets

    def update_curriculum(self):
        # print('BasicTransformDataLoader update_curriculum')
        self.transformation_combination.update_curriculum()
