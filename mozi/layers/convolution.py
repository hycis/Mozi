# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

import theano
from theano.sandbox.cuda.fftconv import conv2d_fft
import theano.tensor as T
# from theano.tensor.signal import downsample
from theano.tensor.signal.pool import pool_2d
from mozi.weight_init import XavierUniformWeight, GaussianWeight
from mozi.layers.template import Template
from mozi.utils.theano_utils import shared_zeros
from theano.tensor.nnet.conv import conv2d
from theano.tensor.conv import conv2d
# from theano.tensor.signal.conv import conv2d

floatX = theano.config.floatX

class Convolution2D(Template):
    def __init__(self, input_channels, filters, kernel_size=(3,3), stride=(1,1),
                 W=None, b=None, weight_init=GaussianWeight(mean=0, std=0.1), border_mode='valid'):
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

        self.W_shape = (self.filters, self.input_channels) + self.kernel_size
        self.W = W
        if self.W is None:
            self.W = weight_init(self.W_shape, name='W')

        self.b = b
        if self.b is None:
            self.b = shared_zeros(shape=(self.filters,), name='b')

        self.params = [self.W, self.b]


    def _train_fprop(self, state_below):
        conv_out = conv2d(state_below, self.W,
                                                  border_mode=self.border_mode,
                                                  subsample=self.stride)
        return conv_out + self.b.dimshuffle('x', 0, 'x', 'x')


    def _layer_stats(self, state_below, layer_output):
        w_max = self.W.max()
        w_min = self.W.min()
        w_mean = self.W.mean()
        w_std = self.W.std()
        return[('filter_max', w_max),
               ('filter_min', w_min),
               ('filter_mean', w_mean),
               ('filter_std', w_std)]



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
            ignore_border:
        '''

        self.poolsize = poolsize
        self.stride = stride
        self.padding = padding
        self.ignore_border = ignore_border
        self.mode = mode

        self.params = []


    def _train_fprop(self, state_below):
        return pool_2d(state_below, ds=self.poolsize, st=self.stride,
                       padding=self.padding, ignore_border=self.ignore_border,
                       mode=self.mode)


class ConvFFT2D(Template):
    def __init__(self, input_channels, filters, stride, kernel_size=(3,3),
        W=None, b=None, weight_init=GaussianWeight(mean=0, std=0.1),
        image_shape=None, border_mode='valid', pad_last_dim=False):
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
        self.border_mode = border_mode
        self.image_shape = image_shape
        self.pad_last_dim = pad_last_dim

        self.W_shape = (self.filters, self.input_channels) + self.kernel_size
        self.W = W
        if self.W is None:
            self.W = weight_init(self.W_shape, name='W')

        self.b = b
        if self.b is None:
            self.b = shared_zeros(shape=(self.filters,), name='b')

        self.params = [self.W, self.b]


    def _train_fprop(self, state_below):
        conv_out = conv2d_fft(state_below, self.W,
                              border_mode=self.border_mode,
                              image_shape=self.image_shape,
                              pad_last_dim=self.pad_last_dim)
        return conv_out + self.b.dimshuffle('x', 0, 'x', 'x')


class SpatialPyramidPooling(Template):
    def __init__(self, levels=[1,2,3], padding=None, ignore_border=False, mode='max'):
        """
        DESCRIPTION:
            This pooling layer describe the method in SPPnet paper
            Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
        PARAM:
            num_levels (list):
                levels in SPP, example [1,2,3]
            ignore_border (bool):
                if x = 5, stride = 2
                True: return 3 = ceil(x / stride)
                False: return 2 = floor(x / stride)
        """

        self.levels = levels
        self.padding = padding
        self.ignore_border = ignore_border
        self.mode = mode

        self.params = []


    def _train_fprop(self, state_below):
        b, c, h, w = state_below.shape
        layer_out = []
        for i in self.levels:
            out = downsample.max_pool_2d(state_below, ds=(h/i, w/i),
                                         st=(h/i, w/i),
                                         padding=self.padding,
                                         ignore_border=self.ignore_border,
                                         mode=self.mode)

            # theano.scan(downsample.max_pool_2d, sequences=[self.levels])
            layer_out.append(out.reshape((b, T.prod(out.shape)/b)))

        return T.concat(layer_out, axis=1)
