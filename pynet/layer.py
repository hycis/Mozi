import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

floatX = theano.config.floatX
theano_rand = RandomStreams(seed=1012)

class Layer(object):
    """
    Abstract Class
    """

    def __init__(self, dim, name, W=None, b=None, dropout_below=None):
        """
        DESCRIPTION:
            This is an abstract layer class
        PARAM:
            dim(int): dimension of the layer
            name(string): name of the layer
            W(tensor variable): Weight of 2D tensor matrix
            b(tensor variable): bias of 2D tensor matrix
            dropout_below: probability of the inputs from the layer below been masked out
        """
        self.dim = dim
        self.name = name
        self.W = W
        self.b = b
        # assert not (dropout_below and blackout_below), 'cannot set both dropout and blackout'
        self.dropout_below = dropout_below
        # self.blackout_below = blackout_below

        if self.W is not None and self.W.name is None:
            self.W.name = 'W_' + self.name
        if self.b is not None and self.b.name is None:
            self.b.name = 'b_' + self.name


    def _test_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _test_fprop.")


    def _train_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _train_fprop.")


    def _mask_state_below(self, state_below):
        if self.dropout_below is not None:
            assert self.dropout_below >= 0. and self.dropout_below <= 1., \
                    'dropout_below is not in range [0,1]'
            state_below = theano_rand.binomial(size=state_below.shape, n=1,
                                               p=(1-self.dropout_below),
                                               dtype=floatX) * state_below
        return state_below


    def _linear_part(self, state_below):
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

        # w_len = T.sqrt((self.W ** 2).sum(axis=0)).astype(floatX)
        # max_length = T.max(w_len).astype(floatX)
        # mean_length = T.mean(w_len).astype(floatX)
        # min_length = T.min(w_len).astype(floatX)
        # max_output = T.max(layer_output).astype(floatX)
        # mean_output = T.mean(T.gt(T.abs_(state_below),0).astype(floatX))
        # min_output = T.min(layer_output).astype(floatX)
        # state_below = self._mask_state_below(state_below)
        #
        # pos = T.mean(T.gt(T.abs_(state_below),0).astype(floatX))
        # if T.gt(pos, 0.4) == 1.0 and T.lt(pos, 0.6) == 1.0:
        #     out = 100
        # else:
        #     out = 0

        # out = T.switch(T.gt(pos, 0.4) , T.switch(T.lt(pos,0.6), 100, 0), 0).astype(floatX)

        # mean_state = T.mean(T.abs_(state_below))
        # test_state = T.gt(mean_state, 0).astype(floatX) and 1.0 or 0.0
        # true_state = T.mean(T.gt(T.abs_(state_below), 0)).astype(floatX)
        return []
        # ('max_col_length', max_length),
        #         ('mean_col_length', mean_length),
        #         ('min_col_length', min_length),
                # ('output_max', max_output),
                # ('output_mean', mean_output),
                # # ('output_min', min_output),
                # ('pos', pos),
                # ('test', out)]
                # ('max_W', T.max(self.W)),
                # ('mean_W', T.mean(self.W)),
                # ('min_W', T.min(self.W)),
                # ('max_b', T.max(self.b)),
                # ('mean_b', T.mean(self.b)),
                # ('min_b', T.min(self.b))]

class Linear(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output

    def _train_fprop(self, state_below):
        state_below = self._mask_state_below(state_below)
        output = self._linear_part(state_below)
        return output


class Sigmoid(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.nnet.sigmoid(output)

    def _train_fprop(self, state_below):
        state_below = self._mask_state_below(state_below)
        output = self._linear_part(state_below)
        return T.nnet.sigmoid(output)


class RELU(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output * (output > 0.)

    def _train_fprop(self, state_below):
        state_below = self._mask_state_below(state_below)
        output = self._linear_part(state_below)
        return output * (output > 0.)


class Softmax(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.nnet.softmax(output)

    def _train_fprop(self, state_below):
        state_below = self._mask_state_below(state_below)
        output = self._linear_part(state_below)
        return T.nnet.softmax(output)


class Tanh(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.tanh(output)

    def _train_fprop(self, state_below):
        state_below = self._mask_state_below(state_below)
        output = self._linear_part(state_below)
        return T.tanh(output)


class Softplus(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.nnet.softplus(output)

    def _train_fprop(self, state_below):
        state_below = self._mask_state_below(state_below)
        output = self._linear_part(state_below)
        return T.nnet.softplus(output)
