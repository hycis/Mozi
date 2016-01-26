
import numpy as np
import theano
from mozi.utils.theano_utils import sharedX


def get_fans(shape):
    '''From keras'''
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


class WeightInitialization(object):

    def __call__(self, dim, name='W'):
        raise NotImplementedError(str(type(self))+" does not implement __call__.")


class GaussianWeight(WeightInitialization):
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, dim, name='W'):
        W_values = np.random.normal(loc=self.mean, scale=self.std, size=dim)
        return sharedX(name=name, value=W_values, borrow=True)


class XavierUniformWeight(WeightInitialization):
    def __call__(self, dim, name='W'):
        fan_in, fan_out = get_fans(dim)
        W_values = np.random.uniform(low = -4 * np.sqrt(6. / (fan_in + fan_out)),
                                     high = 4 * np.sqrt(6. / (fan_in + fan_out)),
                                     size = dim)
        return sharedX(name=name, value=W_values, borrow=True)


class UniformWeight(WeightInitialization):
    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, dim, name='W'):
        W_values = np.random.uniform(low=-self.scale, high=self.scale, size=dim)
        return sharedX(name=name, value=W_values, borrow=True)


class OrthogonalWeight(WeightInitialization):
    def __init__(self, scale=1.1):
        self.scale = scale

    def __call__(self, dim, name='W'):
        ''' From Lasagne
        '''
        flat_shape = (dim[0], np.prod(dim[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(dim)
        return sharedX(name=name, value=self.scale * q[:dim[0],:dim[1]], borrow=True)


class Identity(WeightInitialization):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, dim, name='W'):
        if len(dim) != 2 or dim[0] != dim[1]:
            raise Exception("Identity matrix initialization can only be used for 2D square matrices")
        else:
            return sharedX(self.scale * np.identity(dim[0]))
