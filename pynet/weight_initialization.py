__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"


import numpy as np
import theano

floatX = theano.config.floatX


class WeightInitialization(object):

    def __init__(self, prev_dim, this_dim):
        '''
        PARAMETERS:
            prev_dim(int) : dimension of previous layer
            this_dim(int) : dimension of this layer
        '''
        self.prev_dim = prev_dim
        self.this_dim = this_dim


class GaussianWeight(WeightInitialization):

    def __call__(self, mean=0, std=0.1):
        W_values = np.asarray(np.random.normal(loc = mean, scale = std,
                              size = (self.prev_dim, self.this_dim)),
                              dtype = floatX)
        return theano.shared(value=W_values, borrow=True)


class XavierWeight(WeightInitialization):

    def __call__(self):
        W_values = np.asarray(np.random.uniform(
                            low = -4 * np.sqrt(6. / (self.this_dim + self.prev_dim)),
                            high = 4 * np.sqrt(6. / (self.this_dim + self.prev_dim)),
                            size = (self.prev_dim, self.this_dim)),
                            dtype = floatX)

        return theano.shared(value=W_values, borrow=True)
