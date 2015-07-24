__author__ = "Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"


import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from mozi.utils.theano_utils import shared_zeros
from mozi.weight_init import GaussianWeight
from mozi.layers.template import Template

floatX = theano.config.floatX
theano_rand = MRG_RandomStreams()


class Linear(Template):

    def __init__(self, prev_dim=None, this_dim=None, W=None, b=None,
                 weight_init=GaussianWeight(mean=0, std=0.1)):
        """
        DESCRIPTION:
            This is a fully connected layer
        PARAM:
            prev_dim(int): dimension of previous layer
            this_dim(int): dimension of this layer
            name(string): name of the layer
            W(tensor variable): Weight of 2D tensor matrix
            b(tensor variable): bias of 2D tensor matrix
            params(list): a list of params in layer that can be updated
        """

        self.input_var = T.matrix()
        self.prev_dim = prev_dim
        self.this_dim = this_dim

        self.W = W
        if self.W is None:
            self.W = weight_init((prev_dim, this_dim), name='W_'+self.__class__.__name__)

        self.b = b
        if self.b is None:
            self.b = shared_zeros(shape=this_dim, name='b_'+self.__class__.__name__)

        self.params = [self.W, self.b]


    def _test_fprop(self, state_below):
        """
		DESCRIPTION:
			performs linear transform y = dot(W, state_below) + b
		PARAM:
			state_below: 1d array of inputs from layer below
        """
        return T.dot(state_below, self.W) + self.b


    def _train_fprop(self, state_below):
        """
		DESCRIPTION:
			performs linear transform y = dot(W, state_below) + b
		PARAM:
			state_below: 1d array of inputs from layer below
        """
        return T.dot(state_below, self.W) + self.b


    def _layer_stats(self, state_below, layer_output):
        """
        DESCRIPTION:
            This method is called every batch whereby the examples from test or valid set
            is pass through, the final result will be the mean of all the results from all
            the batches in an epoch from the test set or valid set.
        PARAM:
            layer_output: the output from the layer
        RETURN:
            A list of tuples of [('name_a', var_a), ('name_b', var_b)] whereby var is scalar
        """

        w_len = T.sqrt((self.W ** 2).sum(axis=0))
        max_length = T.max(w_len)
        mean_length = T.mean(w_len)
        min_length = T.min(w_len)
        max_output = T.max(layer_output)
        mean_output = T.mean(T.abs_(layer_output))
        min_output = T.min(layer_output)
        max_state = T.max(state_below)
        mean_state = T.mean(T.abs_(state_below))
        min_state = T.min(state_below)

        return [('max_W', T.max(self.W)),
                ('mean_W', T.mean(self.W)),
                ('min_W', T.min(self.W)),
                ('max_b', T.max(self.b)),
                ('mean_b', T.mean(self.b)),
                ('min_b', T.min(self.b)),
                ('max_layer_output', max_output),
                ('mean_layer_output', mean_output),
                ('min_layer_output', min_output),
                ('max_col_length', max_length),
                ('mean_col_length', mean_length),
                ('min_col_length', min_length),
                ('max_state_below', max_state),
                ('mean_state_below', mean_state),
                ('min_state_below', min_state)]
