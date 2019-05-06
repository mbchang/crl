# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_

.. figure:: /_static/img/stn/FSeq.png

In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__

Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.

One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.
"""
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torchsample
from image_transforms import *

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='../../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=1, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='../../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=1, shuffle=True, num_workers=4)

translate_up = Translate(0.2, 0)()
translate_down = Translate(-0.2, 0)()
translate_left = Translate(0, 0.2)()
translate_right = Translate(0, -0.2)()

translate_up = Translate(0.29, 0)()
translate_down = Translate(-0.29, 0)()
translate_left = Translate(0, 0.29)()
translate_right = Translate(0, -0.29)()

translate_up = Translate(0.2, 0)()
translate_down = Translate(-0.2, 0)()
translate_left = Translate(0, 0.2)()
translate_right = Translate(0, -0.2)()
scale_small = Scale(1.7)()
scale_large = Scale(0.6)()
rotate_right = Rotate(45)()

def test_3c3(train_loader):
    """
        embed image in 64 x 64

        translate_left: 0.2
        scale_large: Zoom 0.6

        translate_left: 0.38
        scale_small: 1.7
    """
    for data, target in train_loader:
        orig = data
        data = place_subimage_in_background((64, 64))(data)

        # TSR
        # transformed1 = translate_left(data)
        # transformed2 = scale_large(transformed1)
        # transformed3 = rotate_right(transformed2)

        # TRS
        # transformed1 = translate_left(data)
        # transformed2 = rotate_right(transformed1)
        # transformed3 = scale_large(transformed2)

        # SRT
        transformed1 = scale_large(data)
        transformed2 = rotate_right(transformed1)
        transformed3 = translate_left(transformed2)

        # RST
        # transformed1 = rotate_right(data)
        # transformed2 = scale_large(transformed1)
        # transformed3 = translate_left(transformed2)

        # STR
        # transformed1 = scale_large(data)
        # transformed2 = translate_left(transformed1)
        # transformed3 = rotate_right(transformed2)

        # RTS
        # transformed1 = rotate_right(data)
        # transformed2 = translate_left(transformed1)
        # transformed3 = scale_large(transformed2)

        orig = convert_image_np(orig)
        data = convert_image_np(data)
        transformed1 = convert_image_np(transformed1)
        transformed2 = convert_image_np(transformed2)
        transformed3 = convert_image_np(transformed3)

        nplots = 5
        f, ax = plt.subplots(1,nplots)
        ax[0].imshow(orig[0])
        ax[1].imshow(data[0])
        ax[2].imshow(transformed1[0])
        ax[3].imshow(transformed2[0])
        ax[4].imshow(transformed3[0])
        for i in range(nplots):
            ax[i].set_axis_off()
        plt.show()


        

        # nplots = 5
        # f, ax = plt.subplots(1,nplots)
        # ax[0].imshow(orig[0])

        # ax[1].imshow(transformed3[0])
        # ax[2].imshow(transformed2[0])
        # ax[3].imshow(transformed1[0])
        # ax[4].imshow(data[0])
        # for i in range(nplots):
        #     ax[i].set_axis_off()
        # plt.show()


def test_3c2(train_loader):
    """
        embed image in 64 x 64

        translate_left: 0.2
        scale_large: Zoom 0.6

        translate_left: 0.38
        scale_small: 1.7
    """
    for data, target in train_loader:
        orig = data
        data = place_subimage_in_background((64, 64))(data)

        # TS
        # transformed1 = translate_left(data)
        # transformed2 = scale_large(transformed1)
        # transformed2 = scale_small(transformed1)

        # ST
        # transformed1 = scale_large(data)
        # transformed1 = scale_small(data)
        # transformed2 = translate_left(transformed1)

        # TR
        # transformed1 = translate_left(data)
        # transformed2 = rotate_right(transformed1)

        # RT
        # transformed1 = rotate_right(data)
        # transformed2 = translate_left(transformed1)

        # SR
        # transformed1 = scale_large(data)
        # transformed1 = scale_small(data)
        # transformed2 = rotate_right(transformed1)

        # RS
        # transformed1 = rotate_right(data)
        # transformed2 = scale_large(transformed1)
        # transformed2 = scale_small(transformed1)

        orig = convert_image_np(orig)
        data = convert_image_np(data)
        transformed1 = convert_image_np(transformed1)
        transformed2 = convert_image_np(transformed2)

        f, ax = plt.subplots(1,4)
        ax[0].imshow(orig[0])
        ax[1].imshow(data[0])
        ax[2].imshow(transformed1[0])
        ax[3].imshow(transformed2[0])
        plt.show()


def test_3c1(train_loader):
    """
        embed image in 64 x 64

        translate_left: 0.2
        scale_large: Zoom 0.6

        translate_left: 0.38
        scale_small: 1.7
    """
    for data, target in train_loader:
        orig = data
        data = place_subimage_in_background((64, 64))(data)

        # T
        # transformed1 = translate_left(data)

        # R
        # transformed1 = rotate_right(data)

        # S
        # transformed1 = scale_large(data)
        transformed1 = scale_small(data)

        # print(orig.size())
        # assert False
        # orig[:, :, 2:10, 2:5] = 1

        orig = convert_image_np(orig)
        data = convert_image_np(data)
        transformed1 = convert_image_np(transformed1)

        f, ax = plt.subplots(1,3)
        ax[0].imshow(orig[0])
        ax[1].imshow(data[0])
        ax[2].imshow(transformed1[0])
        plt.show()


if __name__ == '__main__':
    test_3c3(train_loader)





