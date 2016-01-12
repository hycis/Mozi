import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from mozi.layers.template import Template
from mozi.utils.theano_utils import sharedX

floatX = theano.config.floatX
theano_rand = MRG_RandomStreams()


class Sigmoid(Template):
    def _test_fprop(self, state_below):
        return T.nnet.sigmoid(state_below)

    def _train_fprop(self, state_below):
        return T.nnet.sigmoid(state_below)


class RELU(Template):
    def _test_fprop(self, state_below):
        return state_below * (state_below > 0.)

    def _train_fprop(self, state_below):
        return state_below * (state_below > 0.)


class PRELU(Template):
    def __init__(self, alpha=0.2, **kwargs):
        '''
        y = wx + b
        if y > 0 then z = y else z = alpha * y
        return z
        alpha: the gradient of the slope which is updated by backpropagation
        '''
        super(PRELU, self).__init__(**kwargs)
        alpha = alpha * np.ones(shape=self.dim, dtype=floatX)
        self.alpha = theano.shared(value=alpha, name='PRELU_gradient', borrow=True)
        self.params += [self.alpha]

    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)

    def _train_fprop(self, state_below):
        return state_below * (state_below >= 0) \
        + self.alpha * state_below * (state_below < 0)


class LeakyRELU(Template):
    def __init__(self, alpha=0.01, **kwargs):
        self.alpha = sharedX(alpha)
        self.params = []

    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)

    def _train_fprop(self, state_below):
        return state_below * (state_below >= 0) \
        + self.alpha * state_below * (state_below < 0)


class Noisy_RELU(Template):
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
        return output * (output > self.threshold)

    def _train_fprop(self, state_below):
        if self.batch_count > self.num_batch:
            return state_below * (state_below > self.threshold)

        else:
            self.batch_count += 1
            state_below = state_below + theano_rand.normal(size=state_below.shape, std=self.std, dtype=floatX)
            state_below = state_below * (state_below > self.threshold)
            activity = theano.mean(state_below > 0, axis=0)
            self.activity = self.alpha * activity + (1-self.alpha) * self.activity
            self.threshold += self.threshold_lr * (self.activity - self.sparsity_factor)
            return state_below * (state_below > self.threshold)


class Softmax(Template):
    def _test_fprop(self, state_below):
        return T.nnet.softmax(state_below)

    def _train_fprop(self, state_below):
        return T.nnet.softmax(state_below)


class Tanh(Template):
    def _test_fprop(self, state_below):
        return T.tanh(state_below)

    def _train_fprop(self, state_below):
        return T.tanh(state_below)


class Softplus(Template):
    def _test_fprop(self, state_below):
        return T.nnet.softplus(state_below)

    def _train_fprop(self, state_below):
        return T.nnet.softplus(state_below)
