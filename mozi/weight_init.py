
import numpy as np
import theano

floatX = theano.config.floatX


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
        W_values = np.asarray(np.random.normal(loc = self.mean, scale = self.std,
                              size = dim),
                              dtype = floatX)
        return theano.shared(name=name, value=W_values, borrow=True)


class XavierUniformWeight(WeightInitialization):
    def __call__(self, dim, name='W'):
        fan_in, fan_out = get_fans(dim)
        W_values = np.asarray(np.random.uniform(
                            low = -4 * np.sqrt(6. / (fan_in + fan_out)),
                            high = 4 * np.sqrt(6. / (fan_in + fan_out)),
                            size = dim),
                            dtype = floatX)

        return theano.shared(name=name, value=W_values, borrow=True)


class UniformWeight(WeightInitialization):
    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, dim, name='W'):
        W_values = np.ramdom.uniform(low=-self.scale, high=self.scale, size=dim)
        return theano.shared(name=name, value=W_values, borrow=True)
