
from mozi.layers.template import Template
from mozi.utils.theano_utils import shared_zeros, sharedX
from mozi.weight_init import UniformWeight
import theano.tensor as T
import theano

class BatchNormalization(Template):

    def __init__(self, dim, layer_type, gamma_init=UniformWeight(), short_memory=0.1):
        '''
        REFERENCE:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        PARAMS:
            short_memory: short term memory
                y_t is the latest value, the moving average x_tp1 is calculated as
                x_tp1 = memory * y_t + (1-memory) * x_t, the larger the short term
                memory, the more weight is put on contempory.
            layer_type: fc or conv
            epsilon:
                denominator min value for preventing division by zero in computing std
            dim: for fc layers, shape is the layer dimension, for conv layers,
                shape is the number of feature maps
        '''

        assert layer_type in ['fc', 'conv']
        self.layer_type = layer_type
        self.epsilon = 1e-6
        self.dim = dim
        self.mem = short_memory

        if self.layer_type == 'fc':
            input_shape = (1, dim)
            self.broadcastable = (True, False)
        elif self.layer_type == 'conv':
            input_shape = (1, dim, 1, 1)
            self.broadcastable = (True, False, True, True)

        self.gamma = gamma_init(input_shape, name='gamma')
        self.beta = shared_zeros(input_shape, name='beta')
        self.params = [self.gamma, self.beta]
        self.moving_mean = 0
        self.moving_var = 1



    def _train_fprop(self, state_below):
        if self.layer_type == 'fc':
            miu = state_below.mean(axis=0)
            var = T.mean((state_below - miu)**2, axis=0)
        elif self.layer_type == 'conv':
            miu = state_below.mean(axis=(0,2,3), keepdims=True)
            var = T.mean((state_below - miu)**2, axis=(0,2,3), keepdims=True)

        self.moving_mean += self.mem * miu + (1-self.mem) * self.moving_mean
        self.moving_var += self.mem * var + (1-self.mem) * self.moving_var
        Z = (state_below - self.moving_mean) / T.sqrt(self.moving_var + self.epsilon)
        gamma = T.patternbroadcast(self.gamma, self.broadcastable)
        beta = T.patternbroadcast(self.beta, self.broadcastable)
        return gamma * Z + beta


    def _test_fprop(self, state_below):
        Z = (state_below - self.moving_mean) / T.sqrt(self.moving_var + self.epsilon)
        gamma = T.patternbroadcast(self.gamma, self.broadcastable)
        beta = T.patternbroadcast(self.beta, self.broadcastable)
        return gamma * Z + beta


    def _layer_stats(self, state_below, layer_output):
        return [('moving_mean', T.mean(self.moving_mean)),
                ('moving_std', T.mean(self.moving_var)),
                ('gamma_mean', T.mean(self.gamma)),
                ('beta_mean', T.mean(self.beta)),
                ('gamma_max', T.max(self.gamma))]


# class LRN(Template):
#     """
#     Adapted from pylearn2
#     Local Response Normalization
#     """
#
#     def __init__(self, n=5, alpha=0.0001, beta=0.75, k=2):
#         super(LRN, self).__init__()
#         self.n = n
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k
#         assert self.n % 2 == 1, 'only odd n is supported'
#
#     def _train_fprop(self, state_below):
#         half = self.n / 2
#         sq = T.sqr(state_below)
#         b, ch, r, c = state_below.shape
#         extra_channels = T.alloc(0., b, ch + 2*half, r, c)
#         sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)
#         scale = self.k
#
#         for i in xrange(self.n):
#             scale += self.alpha * sq[:,i:i+ch,:,:]
#
#         scale = scale ** self.beta
#         return state_below / scale
#
#     def _test_fprop(self, state_below):
#         return self._train_fprop(state_below)
