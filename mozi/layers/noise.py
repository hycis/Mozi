
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from mozi.layers.template import Template

floatX = theano.config.floatX
theano_rand = MRG_RandomStreams()

class Dropout(Template):

    def __init__(self, dropout_below=0.5):
        '''
        PARAMS:
            dropout_below(float): probability of the inputs from the layer below been masked out
        '''
        self.dropout_below = dropout_below
        self.params = []


    def _test_fprop(self, state_below):
        """
        DESCRIPTION:
            resize the weight during testing for models trained with dropout.
            The weight will be resized to W' = self.dropout_below * W
        """
        return state_below * (1 - self.dropout_below)


    def _train_fprop(self, state_below):
        """
        DESCRIPTION:
            Applies dropout to the layer during training
        """
        return theano_rand.binomial(size=state_below.shape, n=1,
                                    p=(1-self.dropout_below),
                                    dtype=floatX) * state_below


class MaskOut(Template):

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
        self.params = []

    def _train_fprop(self, state_below):
        return state_below * theano_rand.binomial(size=state_below.shape, n=1, p=(1-self.ratio), dtype=floatX)


class Gaussian(Template):
    """
    Applies gaussian noise to each value of X
    """

    def __init__(self, std=0.1, mean=0):
        self.std = std
        self.mean = mean
        self.params = []

    def _train_fprop(self, state_below):
        return state_below + theano_rand.normal(avg=self.mean, std=self.std, size=state_below.shape, dtype=floatX)


class BlackOut(Template):
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
        self.params = []

    def _train_fprop(self, state_below):
        rd = theano_rand.binomial(size=(state_below.shape[0],), n=1, p=(1-self.ratio), dtype=floatX)
        return state_below * T.shape_padright(rd)


class BatchOut(Template):
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
        self.params = []

    def _train_fprop(self, state_below):
        rd = theano_rand.binomial(size=(1,1), n=1, p=(1-self.ratio), dtype=floatX)
        return state_below * T.patternbroadcast(rd, broadcastable=(True, True))
