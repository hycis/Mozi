
from mozi.layers.template import Template
from mozi.utils.theano_utils import shared_zeros
from mozi.weight_init import UniformWeight
import theano.tensor as T

class BatchNormalization(Template):
    '''
    From keras
    REFERENCE:
        Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            http://arxiv.org/pdf/1502.03167v3.pdf

        mode: 0 -> featurewise normalization
              1 -> samplewise normalization (may sometimes outperform featurewise mode)

        momentum: momentum term in the computation of a running estimate of the mean and std of the data
    '''
    def __init__(self, input_shape, epsilon=1e-6, mode=0, momentum=0.9):
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.mode = mode
        self.momentum = momentum

        self.init = UniformWeight()
        self.gamma = self.init((self.input_shape), name='gamma')
        self.beta = shared_zeros(self.input_shape, name='beta')

        self.running_mean = None
        self.running_std = None

        self.params = [self.gamma, self.beta]


    def _train_fprop(self, state_below):

        if self.mode == 0:
            m = state_below.mean(axis=0)
            # manual computation of std to prevent NaNs
            std = T.mean((state_below-m)**2 + self.epsilon, axis=0) ** 0.5
            X_normed = (state_below - m) / (std + self.epsilon)

            if self.running_mean is None:
                self.running_mean = m
                self.running_std = std
            else:
                self.running_mean *= self.momentum
                self.running_mean += (1-self.momentum) * m
                self.running_std *= self.momentum
                self.running_std += (1-self.momentum) * std

        elif self.mode == 1:
            m = state_below.mean(axis=-1, keepdims=True)
            std = state_below.std(axis=-1, keepdims=True)
            X_normed = (state_below - m) / (std + self.epsilon)

        return self.gamma * X_normed + self.beta


    def _test_fprop(self, state_below):

        if self.mode == 0:
            X_normed = (state_below - self.running_mean) / (self.running_std + self.epsilon)

        elif self.mode == 1:
            m = state_below.mean(axis=-1, keepdims=True)
            std = state_below.std(axis=-1, keepdims=True)
            X_normed = (state_below - m) / (std + self.epsilon)

        return self.gamma * X_normed + self.beta
