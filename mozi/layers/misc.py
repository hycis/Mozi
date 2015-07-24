__author__ = "Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"

"""
Functionality : Define the noise that is to be added to each layer
"""

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from mozi.layers.template import Template

floatX = theano.config.floatX
theano_rand = MRG_RandomStreams()

class Flatten(Template):

    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)

    def _train_fprop(self, state_below):
        size = theano.tensor.prod(state_below.shape) / state_below.shape[0]
        nshape = (state_below.shape[0], size)
        return theano.tensor.reshape(state_below, nshape)
