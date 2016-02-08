from __future__ import absolute_import
import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX
'''
from keras
'''

def asfloatX(X):
    return np.asarray(X, dtype=floatX)

def sharedX(value, dtype=floatX, name=None, borrow=False, **kwargs):
    return theano.shared(np.asarray(value, dtype=dtype), name=name, borrow=borrow, **kwargs)

def shared_zeros(shape, dtype=floatX, name=None, **kwargs):
    return sharedX(np.zeros(shape), dtype=dtype, name=name, **kwargs)

def shared_scalar(val=0., dtype=floatX, name=None, **kwargs):
    return theano.shared(np.cast[dtype](val), **kwargs)

def shared_ones(shape, dtype=floatX, name=None, **kwargs):
    return sharedX(np.ones(shape), dtype=dtype, name=name, **kwargs)

def alloc_zeros_matrix(*dims):
    return T.alloc(np.cast[floatX](0.), *dims)
