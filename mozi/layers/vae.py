
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from mozi.layers.template import Template
from mozi.weight_init import GaussianWeight
from mozi.utils.theano_utils import shared_zeros

floatX = theano.config.floatX
theano_rand = MRG_RandomStreams()

class VariationalAutoencoder(Template):

    def __init__(self, input_dim, bottlenet_dim, z_dim, weight_init=GaussianWeight(mean=0, std=0.01)):

        self.input_dim = input_dim
        self.bottlenet_dim = bottlenet_dim

        # encoder
        self.W_e = weight_init((input_dim, bottlenet_dim), name='W_e')
        self.b_e = shared_zeros(shape=bottlenet_dim, name='b_e')
        self.W_miu = weight_init((bottlenet_dim, z_dim), name='W_miu')
        self.b_miu = shared_zeros(shape=z_dim, name='b_miu')
        self.W_sig = weight_init((bottlenet_dim, z_dim), name='W_sig')
        self.b_sig = shared_zeros(shape=z_dim, name='b_sig')
        # decoder
        self.W1_d = weight_init((z_dim, bottlenet_dim), name='W1_d')
        self.b1_d = shared_zeros(shape=bottlenet_dim, name='b1_d')
        self.W2_d = weight_init((bottlenet_dim, input_dim), name='W2_d')
        self.b2_d = shared_zeros(shape=input_dim, name='b2_d')

        self.params = [self.W_e, self.b_e, self.W_miu, self.b_miu, self.W_sig, self.b_sig,
                       self.W1_d, self.b1_d, self.W2_d, self.b2_d]


    def _train_fprop(self, state_below):
        h_e = T.tanh(T.dot(state_below, self.W_e) + self.b_e)
        miu_e = T.dot(h_e, self.W_miu) + self.b_miu
        logsig_e = 0.5 * (T.dot(h_e, self.W_sig) + self.b_sig)
        eps = theano_rand.normal(avg=0, std=1, size=logsig_e.shape, dtype=floatX)
        z = miu_e + T.exp(logsig_e) * eps
        h_d = T.tanh(T.dot(z, self.W1_d) + self.b1_d)
        y = T.nnet.sigmoid(T.dot(h_d, self.W2_d) + self.b2_d)
        return y, miu_e, logsig_e


    def _layer_stats(self, state_below, layer_output):
        y, miu, logsig = layer_output
        return [('W_miu', self.W_miu.mean()),
                ('W_e', self.W_e.mean()),
                ('logsig', logsig.mean()),
                ('ymean,', y.mean()),
                ('miu', miu.mean())]
