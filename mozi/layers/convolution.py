# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from mozi.weight_init import XavierUniformWeight, GaussianWeight
from mozi.layers.template import Template
from mozi.utils.theano_utils import shared_zeros

floatX = theano.config.floatX

class Convolution2D(Template):
    def __init__(self, input_channels, filters, kernel_size=(3,3),
        stride=(1,1), W=None, b=None, weight_init=GaussianWeight(mean=0, std=0.1),
        image_shape=None, border_mode='valid'):
        '''
        PARAM:
            border_mode: (from theano)
                valid: only apply filter to complete patches of the image. Generates
                output of shape: image_shape - filter_shape + 1
                full: zero-pads image to multiple of filter shape to generate output
                of shape: image_shape + filter_shape - 1
        '''
        self.input_channels = input_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.border_mode = border_mode
        self.image_shape = image_shape

        self.W_shape = (self.filters, self.input_channels) + self.kernel_size
        self.W = W
        if self.W is None:
            self.W = weight_init(self.W_shape, name='W_'+self.__class__.__name__)

        self.b = b
        if self.b is None:
            self.b = shared_zeros(shape=(self.filters,), name='b_'+self.__class__.__name__)

        self.params = [self.W, self.b]


    def _train_fprop(self, state_below):
        conv_out = theano.tensor.nnet.conv.conv2d(state_below, self.W,
                                                  border_mode=self.border_mode,
                                                  subsample=self.stride,
                                                  image_shape=self.image_shape)
        return conv_out + self.b.dimshuffle('x', 0, 'x', 'x')


    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)


class Pooling2D(Template):
    def __init__(self, poolsize=(2, 2), stride=None, padding=(0,0),
                 ignore_border=True, mode='max'):
        '''
        DESCRIPTION:
            pooling layer
        PARAM:
            stride: two-dimensional tuple (a, b), the separation horizontally a
                or vertically b between two pools
            padding: pad zeros to the border of the feature map
            mode: max | sum | average_inc_pad | average_exc_pad
        '''

        self.poolsize = poolsize
        self.stride = stride
        self.padding = padding
        self.ignore_border = ignore_border
        self.mode = mode

        self.params = []


    def _train_fprop(self, state_below):
        return downsample.max_pool_2d(state_below, ds=self.poolsize,
                                      st=self.stride, padding=self.padding,
                                      ignore_border=self.ignore_border,
                                      mode=self.mode)


    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)
