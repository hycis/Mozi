__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"


import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

floatX = theano.config.floatX
theano_rand = MRG_RandomStreams()

class Layer(object):
    """
    Abstract Class
    """

    def __init__(self, dim, name, W=None, b=None, dropout_below=None, noise=None, blackout_below=None):
        """
        DESCRIPTION:
            This is an abstract layer class
        PARAM:
            dim(int): dimension of the layer
            name(string): name of the layer
            W(tensor variable): Weight of 2D tensor matrix
            b(tensor variable): bias of 2D tensor matrix
            dropout_below(float): probability of the inputs from the layer below been masked out
            params(list): a list of params in layer that can be updated
        """
        self.dim = dim
        self.name = name
        self.W = W
        self.b = b
        assert not (dropout_below and blackout_below), 'cannot set both dropout and blackout'
        self.dropout_below = dropout_below
        self.noise = noise
        self.blackout_below = blackout_below

        # any params from the layer that needs to be updated
        # by backpropagation can be put inside self.params list
        self.params = []

        if self.W is not None and self.W.name is None:
            self.W.name = 'W_' + self.name
        if self.b is not None and self.b.name is None:
            self.b.name = 'b_' + self.name


    def _test_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _test_fprop.")


    def _train_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _train_fprop.")

    def _apply_noise(self, state_below):
        """
        DESCRIPTION:
            Adds noise to the layer during training
        """
        if self.noise:
            state_below = self.noise.apply(state_below)
        return state_below

    def _dropout_below(self, state_below):
        """
        DESCRIPTION:
            Applies dropout to the layer during training
        """
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

    def _dropout_fprop(self, state_below):
        """
        DESCRIPTION:
            resize the weight during testing for models trained with dropout.
            The weight will be resized to W' = self.dropout_below * W
        """
        return T.dot(state_below, (1-self.dropout_below) * self.W) + self.b

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


        # test_state = T.gt(mean_state, 0).astype(floatX) and 1.0 or 0.0
        # true_state = T.mean(T.gt(T.abs_(state_below), 0)).astype(floatX)
        # activity = T.mean(T.gt(layer_output, 0) * 1.0)
        # return [('activity', activity.astype(floatX)),
        # ('max_col_length', max_length),
        # ('mean_col_length', mean_length),
        # ('min_col_length', min_length),
        # ('output_max', max_output),
        # ('output_mean', mean_output),
        # ('output_min', min_output)]
        # ('pos', pos),
        # ('test', out)]

        # w100 = self.W.sum(axis=1)[100].astype(floatX)
        # w200 = self.W.sum(axis=1)[200].astype(floatX)
        # w300 = self.W.sum(axis=1)[300].astype(floatX)
        # w_len100 = T.sqrt((self.W ** 2).sum(axis=1)[100]).astype(floatX)
        # w_len200 = T.sqrt((self.W ** 2).sum(axis=1)[200]).astype(floatX)
        # w_len300 = T.sqrt((self.W ** 2).sum(axis=1)[300]).astype(floatX)
        # return [('w_sum_axis_1', self.W.sum(axis=1).shape[0].astype(floatX)),
        # ('W_row100', w100),('W_row200', w200),('W_row300', w300),
        # ('len100', w_len100), ('len200', w_len200), ('len300', w_len300),
        return [('max_W', T.max(self.W)),
                ('mean_W', T.mean(self.W)),
                ('min_W', T.min(self.W)),
                ('max_b', T.max(self.b)),
                ('mean_b', T.mean(self.b)),
                ('min_b', T.min(self.b)),
                ('output_max', max_output),
                ('output_mean', mean_output),
                ('output_min', min_output),
                ('max_col_length', max_length),
                ('mean_col_length', mean_length),
                ('min_col_length', min_length),
                ('max_state', max_state),
                ('mean_state', mean_state),
                ('min_state', min_state)]
        # return []

class Linear(Layer):
    def _test_fprop(self, state_below):
        if self.dropout_below:
            return self._dropout_fprop(state_below)
        else:
            return self._linear_part(state_below)

    def _train_fprop(self, state_below):
        state_below = self._dropout_below(state_below)
        state_below = self._apply_noise(state_below)
        output = self._linear_part(state_below)
        return output


class Sigmoid(Linear):
    def _test_fprop(self, state_below):
        output = super(Sigmoid, self)._test_fprop(state_below)
        return T.nnet.sigmoid(output)

    def _train_fprop(self, state_below):
        output = super(Sigmoid, self)._train_fprop(state_below)
        return T.nnet.sigmoid(output)


class RELU(Linear):
    def _test_fprop(self, state_below):
        output = super(RELU, self)._test_fprop(state_below)
        return output * (output > 0.)

    def _train_fprop(self, state_below):
        output = super(RELU, self)._train_fprop(state_below)
        return output * (output > 0.)


    def _layer_stats(self, state_below, layer_output):
        rlist = []

        # sparsity = T.gt(layer_output,0).sum().astype(floatX) / (layer_output.shape[0] * layer_output.shape[1])

        activity = T.mean(T.gt(layer_output, 0) * 1.0)
        # return [('activity', activity.astype(floatX)),

        # rlist.append(('threshold_mean', T.mean(self.threshold).astype(floatX)))
        rlist.append(('activity', activity.astype(floatX)))
        # rlist.append(('threshold_max', T.max(self.threshold).astype(floatX)))
        # rlist.append(('threshold_min', T.min(self.threshold).astype(floatX)))
        # rlist.append(('std_prod', T.prod(layer_output.std(axis=0)).astype(floatX)))
        # rlist.append(('sparsity', sparsity))
        rlist.extend(super(RELU, self)._layer_stats(state_below, layer_output))
        return rlist

class SoftRELU(Linear):
    def __init__(self, threshold=0., **kwargs):
        '''
        threshold: the threshold of z = max(wx+b, threshold) which will be updated
        by backpropagation.
        '''
        super(Linear, self).__init__(**kwargs)
        threshold = threshold * np.ones(shape=self.dim, dtype=floatX)
        self.threshold = theano.shared(value=threshold, name='SoftRELU_Threshold', borrow=True)
        # T.patternbroadcast(self.threshold, broadcastable=(True, False))
        self.params = [self.threshold]

    def _test_fprop(self, state_below):
        output = super(SoftRELU, self)._test_fprop(state_below)
        return output * (output > self.threshold) + self.threshold * (output <= self.threshold)

    def _train_fprop(self, state_below):
        output = super(SoftRELU, self)._train_fprop(state_below)
        return output * (output > self.threshold) + self.threshold * (output <= self.threshold)

    def _layer_stats(self, state_below, layer_output):
        rlist = []

        # sparsity = T.gt(layer_output,0).sum().astype(floatX) / (layer_output.shape[0] * layer_output.shape[1])

        activity = T.mean(T.gt(layer_output, self.threshold) * 1.0)
        # return [('activity', activity.astype(floatX)),

        rlist.append(('threshold_mean', T.mean(self.threshold).astype(floatX)))
        rlist.append(('activity', activity.astype(floatX)))
        rlist.append(('threshold_max', T.max(self.threshold).astype(floatX)))
        rlist.append(('threshold_min', T.min(self.threshold).astype(floatX)))
        rlist.append(('std_prod', T.prod(layer_output.std(axis=0)).astype(floatX)))
        # rlist.append(('sparsity', sparsity))
        return rlist



class Noisy_RELU(Linear):
    def __init__(self, sparsity_factor=0.1, threshold_lr=0.01, alpha=0.01, std=0.1, num_batch=10000, **kwargs):
        '''
         sparsityFactor: the micro sparsity of signals through each neuron
         threshold_lr: the learning rate of learning the optimum threshold for each neuron
                       so that the activeness of the neuron approaches sparsityFactor
         alpha_range: {start_weight, num_batches, final_weight} for setting the weight on the
                      contemporary sparsity when calculating the mean sparsity over many batches.
                      For the start, it will place more weight on the contemporary, but as more
                      epoch goes through, the weight on contemporary batch should decrease, so
                      that mean_sparsity will be more stable.
        std: the standard deviation of the noise
        '''
        super(Noisy_RELU, self).__init__(**kwargs)
        self.sparsity_factor = sparsity_factor
        self.threshold_lr = threshold_lr
        self.alpha = alpha
        self.std = std
        self.num_batch = num_batch
        self.threshold = 0.
        self.activity = 0.
        self.batch_count = 0


    def _test_fprop(self, state_below):
        output = super(Noisy_RELU, self)._test_fprop(state_below)
        return output * (output > self.threshold)

    def _train_fprop(self, state_below):
        output = super(Noisy_RELU, self)._train_fprop(state_below)

        if self.batch_count > self.num_batch:
            return output * (output > self.threshold)

        else:
            self.batch_count += 1
            output = output + theano_rand.normal(size=output.shape, std=self.std, dtype=floatX)
            output = output * (output > self.threshold)
            activity = theano.mean(output > 0, axis=0)
            self.activity = self.alpha * activity + (1-self.alpha) * self.activity
            self.threshold += self.threshold_lr * (self.activity - self.sparsity_factor)
            return output * (output > self.threshold)

    def _layer_stats(self, state_below, layer_output):
        max_act = theano.max(self.activity)
        min_act = theano.min(self.activity)
        mean_act = theano.mean(self.activity)
        max_thresh = theano.max(self.threshold)
        min_thresh = theano.min(self.threshold)
        mean_thresh = theano.mean(self.threshold)

        ls1 = super(Noisy_RELU, self)._layer_stats(self, state_below, layer_output)
        ls2 = [('max_act', max_act),
                ('min_act', min_act),
                ('mean_act', mean_act),
                ('max_thresh', max_thresh),
                ('min_thresh', min_thresh),
                ('mean_thresh', mean_thresh)]

        return ls1 + ls2


class Softmax(Linear):
    def _test_fprop(self, state_below):
        output = super(Softmax, self)._test_fprop(state_below)
        return T.nnet.softmax(output)

    def _train_fprop(self, state_below):
        output = super(Softmax, self)._train_fprop(state_below)
        return T.nnet.softmax(output)

class Sigmoid10X(Linear):

    def _test_fprop(self, state_below):
        output = super(Sigmoid10X, self)._test_fprop(state_below)
        return 10 * T.nnet.sigmoid(output)

    def _train_fprop(self, state_below):
        output = super(Sigmoid10X, self)._train_fprop(state_below)
        return 10 * T.nnet.sigmoid(output)


class Tanh(Linear):
    def _test_fprop(self, state_below):
        output = super(Tanh, self)._test_fprop(state_below)
        return T.tanh(output)

    def _train_fprop(self, state_below):
        output = super(Tanh, self)._train_fprop(state_below)
        return T.tanh(output)

class Tanh5X(Linear):
    def _test_fprop(self, state_below):
        output = super(Tanh5X, self)._test_fprop(state_below)
        return 5.0 * T.tanh(output / 5.0)

    def _train_fprop(self, state_below):
        output = super(Tanh5X, self)._train_fprop(state_below)
        return 5.0 * T.tanh(output / 5.0)

class Tanhkx(Linear):
    def __init__(self, k=10., **kwargs):
        '''

        '''
        super(Tanhkx, self).__init__(**kwargs)
        self.k = theano.shared(value=np.asarray(k, dtype=floatX), name='Tanhkx_k')
        self.params = [self.k]

    def _test_fprop(self, state_below):
        output = super(Tanhkx, self)._test_fprop(state_below)
        return self.k * T.tanh(output / self.k)

    def _train_fprop(self, state_below):
        output = super(Tanhkx, self)._train_fprop(state_below)
        return self.k * T.tanh(output / self.k)

    def _layer_stats(self, state_below, layer_output):
        return [('k', self.k)]


class Softplus(Linear):
    def _test_fprop(self, state_below):
        output = super(Softplus, self)._test_fprop(state_below)
        return T.nnet.softplus(output)

    def _train_fprop(self, state_below):
        output = super(Softplus, self)._train_fprop(state_below)
        return T.nnet.softplus(output)
