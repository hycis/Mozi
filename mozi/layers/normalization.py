
from mozi.layers.template import Template
from mozi.utils.theano_utils import shared_zeros
from mozi.weight_init import UniformWeight
import theano.tensor as T
import theano
floatX = theano.config.floatX

class BatchNormalization(Template):

    def __init__(self, input_shape, epsilon=1e-6, mode=0, gamma_init=UniformWeight(), short_memory=0.9):
        '''
        REFERENCE:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                                 http://arxiv.org/pdf/1502.03167v3.pdf
        PARAMS:
            short_memory: short term memory
                y_t is the latest value, the moving average x_tp1 is calculated as
                x_tp1 = memory * y_t + (1-memory) * x_t, the larger the short term
                memory, the more weight is put on contempory.
            epsilon:
                denominator min value for preventing division by zero in computing std
        '''
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.mem = short_memory

        self.gamma = gamma_init(self.input_shape, name='gamma')
        self.beta = shared_zeros(self.input_shape, name='beta')

        self.moving_mean = 0
        self.moving_std = 0

        self.params = [self.gamma, self.beta]


    def _train_fprop(self, state_below):
        miu = state_below.mean(axis=0)
        std = T.std(state_below, axis=0)
        self.moving_mean += self.mem * miu + (1-self.mem) * self.moving_mean
        self.moving_std += self.mem * std + (1-self.mem) * self.moving_std
        Z = (state_below - self.moving_mean) / (self.moving_std + self.epsilon)
        return self.gamma * Z + self.beta


    def _test_fprop(self, state_below):
        Z = (state_below - self.moving_mean) / (self.moving_std + self.epsilon)
        return self.gamma * Z + self.beta


    def _layer_stats(self, state_below, layer_output):
        return [('moving_mean', self.moving_mean),
                ('moving_std', self.moving_std),
                ('gamma_mean', T.mean(self.gamma))]


class LRN(Template):
    """
    Adapted from pylearn2
    Local Response Normalization
    """

    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=2):
        super(LRN, self).__init__()
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        assert self.n % 2 == 1, 'only odd n is supported'

    def _train_fprop(self, state_below):
        half = self.n / 2
        sq = T.sqr(state_below)
        b, ch, r, c = state_below.shape
        extra_channels = T.alloc(0., b, ch + 2*half, r, c)
        sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)
        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[:,i:i+ch,:,:]

        scale = scale ** self.beta
        return state_below / scale

    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)
