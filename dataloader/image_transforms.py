import numpy as np
import torch
import torchsample
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

""" All transforms are going to operate on (channels, H, W)
    We will manually make them batch compatible.
    All functions are transforms that take in an image and output an image
"""

def batch_compatible(transform):
    """
        Takes a transform function and returns 
        another function that applies the transform
        based on the number of dimension of the data
    """
    def handle_batch(data):
        if data.dim() == 3:
            return transform(data)
        elif data.dim() == 4:

            transformed = map(transform, data)

            if isinstance(transformed[0], Variable) or isinstance(transformed[0], torch.Tensor) or isinstance(transformed[0], torch.cuda.FloatTensor):
                return torch.stack(transformed)
            elif isinstance(transformed[0], np.ndarray):
                return transformed
            else:
                assert False
        else:
            assert False
    batch_compatible_transform = lambda data: handle_batch(data)
    return batch_compatible_transform

def convert_image_np(inp, normalize=False):
    """Convert a Tensor to numpy image."""
    def convert(inp):
        if inp.dim() == 3:
            inp = inp.numpy().transpose((1, 2, 0))
        elif inp.dim() == 4:
            inp = inp.numpy().transpose((0, 2, 3, 1))
        else:
            assert False
        if inp.shape[-1] == 1:
            inp = np.repeat(inp, 3, axis=-1)
        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp
    return batch_compatible(convert)(inp)


######################################################

class Transform(object):
    def __init__(self, cuda):
        super(Transform, self).__init__()
        self.cuda = cuda

    def __call__(self):
        raise NotImplementedError

    def invert(self):
        raise NotImplementedError

class RandomRotate(Transform):
    """ This is a transformation _generator_

        every time you call __call__, self.deg is set. If you call __call__
        again, self.deg will be reset.

        if self.deg is set, then you can also generate the inverse transform
        which will rotate it back. But if you reset self.deg, then the 
        inverse transform will also be reset.
    """
    def __init__(self, lo, hi, cuda=False):
        super(RandomRotate, self).__init__(cuda)
        self.lo = lo
        self.hi = hi

    def __call__(self):
        self.deg = np.random.uniform(self.lo, self.hi)
        # transform = torchsample.transforms.Rotate(self.deg, cuda=self.cuda)
        transform = torchsample.transforms.Rotate(self.deg)
        return batch_compatible(transform)

    def invert(self):
        inverse_deg = -self.deg
        # transform = torchsample.transforms.Rotate(inverse_deg, cuda=self.cuda)
        transform = torchsample.transforms.Rotate(inverse_deg)
        return batch_compatible(transform)

    def __eq__(self, other):
        return self.deg == other.deg

class Rotate(Transform):
    def __init__(self, deg, cuda=False):
        super(Rotate, self).__init__(cuda)
        self.deg = deg

    def __call__(self):
        # transform = torchsample.transforms.Rotate(self.deg, cuda=self.cuda)
        transform = torchsample.transforms.Rotate(self.deg)
        return batch_compatible(transform)

    def invert(self):
        inverse_deg = -self.deg
        # transform = torchsample.transforms.Rotate(inverse_deg, cuda=self.cuda)
        transform = torchsample.transforms.Rotate(inverse_deg)
        return batch_compatible(transform)

    def __eq__(self, other):
        return self.deg == other.deg

    def get_parameter(self):
        return {'deg': self.deg}

    def get_inverse_parameter(self):
        return {'deg': 1.0/self.deg}

class CurriculumRotate(Rotate):
    def __init__(self, curriculum, cuda=False):
        self.curriculum = curriculum
        self.curr_index = 0
        super(CurriculumRotate, self).__init__(deg=self.curriculum[self.curr_index], cuda=cuda)

    def update_curriculum(self):
        self.curr_index += 1
        if self.curr_index >= len(self.curriculum)-1:
            self.curr_index = len(self.curriculum)-1
            # print('Curriculum reached saturation')
        self.deg = self.curriculum[self.curr_index]

class RandomScale(Transform):
    def __init__(self, lo, hi, cuda=False):
        super(RandomScale, self).__init__(cuda)
        self.lo = lo
        self.hi = hi

    def __call__(self):
        self.scale = np.random.rand()*(self.hi-self.lo)+self.lo
        # transform = torchsample.transforms.Zoom(self.scale, cuda=self.cuda)
        transform = torchsample.transforms.Zoom(self.scale)
        return batch_compatible(transform)

    def invert(self):
        inverse_scale = 1.0/self.scale
        # transform = torchsample.transforms.Zoom(inverse_scale, cuda=self.cuda)
        transform = torchsample.transforms.Zoom(inverse_scale)
        return batch_compatible(transform)

    def __eq__(self, other):
        return self.lo == other.lo and self.hi == other.hi

class Scale(Transform):
    def __init__(self, scale, cuda=False):
        super(Scale, self).__init__(cuda)
        self.scale = scale

    def __call__(self):
        # transform = torchsample.transforms.Zoom(self.scale, cuda=self.cuda)
        transform = torchsample.transforms.Zoom(self.scale)
        return batch_compatible(transform)

    def invert(self):
        inverse_scale = 1.0/self.scale
        # transform = torchsample.transforms.Zoom(inverse_scale, cuda=self.cuda)
        transform = torchsample.transforms.Zoom(inverse_scale)
        return batch_compatible(transform)

    def __eq__(self, other):
        return self.scale == other.scale

    def get_parameter(self):
        return {'scale': self.scale}

    def get_inverse_parameter(self):
        return {'scale': 1.0/self.scale}

class CurriculumScale(Scale):
    def __init__(self, curriculum, cuda=False):
        self.curriculum = curriculum
        self.curr_index = 0
        super(CurriculumScale, self).__init__(scale=self.curriculum[self.curr_index], cuda=cuda)

    def update_curriculum(self):
        self.curr_index += 1
        if self.curr_index >= len(self.curriculum)-1:
            self.curr_index = len(self.curriculum)-1
        self.scale = self.curriculum[self.curr_index]

class RandomTranslate(Transform):
    def __init__(self, hrange, vrange, cuda=False):
        super(RandomTranslate, self).__init__(cuda)
        self.hrange = hrange
        self.vrange = vrange

    def __call__(self):
        self.h = np.random.uniform(-self.hrange, self.hrange)
        self.v = np.random.uniform(-self.vrange, self.vrange)
        # transform = torchsample.transforms.Translate((self.h, self.v), cuda=self.cuda)
        transform = torchsample.transforms.Translate((self.h, self.v))
        return batch_compatible(transform)

    def invert(self):
        inverse_h = -self.h
        inverse_v = -self.v
        # transform = torchsample.transforms.Translate((inverse_h, inverse_v), cuda=self.cuda)
        transform = torchsample.transforms.Translate((inverse_h, inverse_v))
        return batch_compatible(transform)

    def __eq__(self, other):
        return self.hrange == other.hrange and self.vrange == other.vrange

class Translate(Transform):
    def __init__(self, h, v, cuda=False):
        super(Translate, self).__init__(cuda)
        self.h = h
        self.v = v

    def __call__(self):
        # transform = torchsample.transforms.Translate((self.h, self.v), cuda=self.cuda)
        transform = torchsample.transforms.Translate((self.h, self.v))
        return batch_compatible(transform)

    def invert(self):
        inverse_h = -self.h
        inverse_v = -self.v
        # transform = torchsample.transforms.Translate((inverse_h, inverse_v), cuda=self.cuda)
        transform = torchsample.transforms.Translate((inverse_h, inverse_v))
        return batch_compatible(transform)

    def __eq__(self, other):
        return self.h == other.h and self.v == other.v

    def get_parameter(self):
        return {'h': self.h, 'v': self.v}

    def get_inverse_parameter(self):
        return {'h': -self.h, 'v': -self.v}

class CurriculumTranslate(Translate):
    def __init__(self, curriculum, cuda=False):
        self.curriculum = curriculum
        self.curr_index = 0
        super(CurriculumTranslate, self).__init__(
            h=self.curriculum[self.curr_index][0], 
            v=self.curriculum[self.curr_index][1], cuda=cuda)

    def update_curriculum(self):
        self.curr_index += 1
        if self.curr_index >= len(self.curriculum)-1:
            self.curr_index = len(self.curriculum)-1
        self.h = self.curriculum[self.curr_index][0]
        self.v = self.curriculum[self.curr_index][1]

def flip_vertical(cuda=False):
    # hardcode this myself
    def transform(data):
        # data: (B, H, W)
        assert data.dim() == 3
        inv_index = torch.arange(data.size(1)-1, -1, -1).long()
        if cuda: inv_index = inv_index.cuda()
        data = data[:, inv_index]
        return data

    return batch_compatible(transform)

def flip_horizontal(cuda=False):
    # hardcode this myself
    def transform(data):
        # data: (B, H, W)
        assert data.dim() == 3
        inv_index = torch.arange(data.size(2)-1, -1, -1).long()
        if cuda: inv_index = inv_index.cuda()
        data = data[:, :, inv_index]
        return data

    return batch_compatible(transform)

class RandomFlip(Transform):
    def __init__(self, cuda=False):
        super(RandomFlip, self).__init__(cuda)

    def __call__(self):
        self.p = np.random.rand()
        if self.p < 0.5:
            transform = flip_vertical(cuda=self.cuda)
        else:
            transform = flip_horizontal(cuda=self.cuda)
        return batch_compatible(transform)

    def invert(self):
        inverse_p = self.p
        if self.p < 0.5:
            transform = flip_vertical(cuda=self.cuda)
        else:
            transform = flip_horizontal(cuda=self.cuda)
        return batch_compatible(transform)

class Invert(Transform):
    def __init__(self, cuda=False):
        super(Invert, self).__init__(cuda)

    def __call__(self):
        transform = lambda data: 1 - data
        return batch_compatible(transform)

    def invert(self):
        transform = lambda data: 1 - data
        return batch_compatible(transform)

class Identity(Transform):
    def __init__(self, cuda=False):
        super(Identity, self).__init__(cuda)

    def __call__(self):
        transform = lambda x: x
        return batch_compatible(transform)

    def invert(self):
        transform = lambda x: x
        return batch_compatible(transform)

    def __eq__(self, other):
        return True

    def get_parameter(self):
        return {}

    def get_inverse_parameter(self):
        return {}

class CurriculumIdentity(Identity):
    def __init__(self, cuda=False):
        self.curriculum = None
        super(CurriculumIdentity, self).__init__(cuda=cuda)

    def update_curriculum(self):
        pass

def place_subimage_in_background(bkdg_dim, rand=False):
    def transform(subimage):
        assert subimage.dim() == 3
        subimage_channels, subimage_height, subimage_width = subimage.size()
        assert subimage_height < bkdg_dim[0] and subimage_width < bkdg_dim[1]
        bkgd = torch.zeros((subimage_channels, bkdg_dim[0], bkdg_dim[1]))
        # get limits
        from_top_limit = bkdg_dim[0]-subimage_height+1
        from_left_limit = bkdg_dim[1]-subimage_width+1
        if rand:
            top = np.random.randint(0, from_top_limit)
            left = np.random.randint(0, from_left_limit)
        else:
            top = from_top_limit/2
            left = from_left_limit/2
        # place subimage inside background
        bkgd[:, top:top+subimage_height, left:left+subimage_width] += subimage
        return bkgd
    return batch_compatible(transform)