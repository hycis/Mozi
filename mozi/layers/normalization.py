
from mozi.layers.template import Template
from mozi.utils.theano_utils import shared_zeros, sharedX
from mozi.weight_init import UniformWeight
import theano.tensor as T
import theano
floatX = theano.config.floatX

class BatchNormalization(Template):

    def __init__(self, input_shape, gamma_init=UniformWeight(), short_memory=0.9):
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
        assert isinstance(input_shape, (tuple, list))
        assert len(input_shape) == 1 or 3, 'batchnorm only applies to 1d and 3d(image) dataset currently'
        self.epsilon = 1e-6
        self.input_shape = input_shape
        self.mem = short_memory

        init_shape = self.input_shape
        if len(input_shape) == 3:
            c, h, w = self.input_shape
            init_shape = (c,)
            # self.gamma = gamma_init(init_shape, name='gamma',  broadcastable=[True, False, True, True])
            # self.beta = shared_zeros(init_shape, name='beta', broadcastable=[True, False, True, True])
        # else:
        print 'init_shape', init_shape
        self.gamma = gamma_init(init_shape, name='gamma')
        self.beta = shared_zeros(init_shape, name='beta')

        self.moving_mean = 0
        self.moving_var = 1

        self.params = [self.gamma, self.beta]


    def _train_fprop(self, state_below):
        b = state_below.shape[0]
        if len(self.input_shape) == 3:
            c, h, w = self.input_shape
            state_below = state_below.reshape((b*c, h*w))
            miu = state_below.mean(axis=1, keepdims=True) # (b*c,1)
            var = T.mean((state_below - miu) ** 2, axis=1, keepdims=True)
        else:
            miu = state_below.mean(axis=0, keepdims=True) #(num_fea,)
            var = T.mean((state_below - miu) ** 2, axis=0, keepdims=True)
        self.moving_mean += self.mem * miu + (1-self.mem) * self.moving_mean
        self.moving_var += self.mem * var + (1-self.mem) * self.moving_var
        Z = (state_below - self.moving_mean) / T.sqrt(self.moving_var + self.epsilon)

        if len(self.input_shape) == 3:
            Z = Z.reshape((b,c,h,w)).swapaxes(1,3) # (b,w,h,c)
            Z = Z.reshape((b*w*h, c))
            out = self.gamma * Z + self.beta
            Z = Z.reshape((b,w,h,c))
            Z = Z.swapaxes(1,3)
            return out.reshape((b, c, h, w))

        else:
            # (f,) * (b, f)
            return self.gamma * Z + self.beta

        # if len(self.input_shape) == 3:
        #     out = out.reshape((b, c, h, w))
        # return out


    def _test_fprop(self, state_below):
        b = state_below.shape[0]
        if len(self.input_shape) == 3:
            c, h, w = self.input_shape
            state_below = state_below.reshape((b*c, h*w))

        # else:
        Z = (state_below - self.moving_mean) / T.sqrt(self.moving_var + self.epsilon)

        if len(self.input_shape) == 3:
            Z = Z.reshape((b*h*w, c))
            out = self.gamma * Z + self.beta
            return out.reshape((b, c, h, w))

        else:
            # (f,) * (b, f)
            return self.gamma * Z + self.beta


    def _layer_stats(self, state_below, layer_output):
        return [('moving_mean', T.mean(self.moving_mean)),
                ('moving_var', T.mean(self.moving_var)),
                ('gamma_mean', T.mean(self.gamma)),
                ('beta_mean', T.mean(self.beta)),
                ('memory', sharedX(self.mem))]


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
