__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"

"""
Functionality : Define the noise that is to be added to each layer
"""

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

floatX = theano.config.floatX
theano_rand = MRG_RandomStreams()

class Noise(object):
    """
    This is an abstract class for applying noise to each layer
    """

    def apply(self, X):
        """
        DESCRIPTION:
            This method applies noise to X and return a noisy X
        PARAM:
            X : 2d numpy array of dimension number of examples by number of dimensions
        """
        raise NotImplementedError(str(type(self))+" does not implement an apply method.")



class MaskOut(Noise):

    """
    This noise masked out a portion of the dimension from each example
    """

    def __init__(self, ratio=0.5):
        """
        PARAM:
            ratio : float
                The portion of the inputs that is masked out
        """
        self.ratio = ratio

    def apply(self, X):
        return X * theano_rand.binomial(size=X.shape, n=1, p=(1-self.ratio), dtype=floatX)


class Gaussian(Noise):
    """
    Applies gaussian noise to each value of X
    """

    def __init__(self, std=0.1, mean=0):
        self.std = std
        self.mean = mean

    def apply(self, X):
        return X + theano_rand.normal(avg=self.mean, std=self.std, size=X.shape, dtype=floatX)


class BlackOut(Noise):
    """
    This noise masked out a random example in a dataset,
    adding noise in the time dimension
    """

    def __init__(self, ratio=0.5):
        """
        PARAM:
            ratio : float
                The portion of the examples that is masked out
        """
        self.ratio = ratio

    def apply(self, X):
        rd = theano_rand.binomial(size=(X.shape[0],), n=1, p=(1-self.ratio), dtype=floatX)
        return X * T.shape_padright(rd)

class BatchOut(Noise):
    """
    This noise masked out a random batch in an epoch,
    adding noise in the time dimension
    """

    def __init__(self, ratio=0.5):
        """
        PARAM:
            ratio : float
                The portion of the batch that is masked out
        """
        self.ratio = ratio

    def apply(self, X):
        rd = theano_rand.binomial(size=(1,1), n=1, p=(1-self.ratio), dtype=floatX)
        return X * T.patternbroadcast(rd, broadcastable=(True, True))
