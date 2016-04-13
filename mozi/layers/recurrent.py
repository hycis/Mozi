
from mozi.utils.theano_utils import shared_zeros, alloc_zeros_matrix, shared_ones
from mozi.layers.template import Template
from mozi.weight_init import OrthogonalWeight, GaussianWeight, Identity
import theano.tensor as T
import theano


class LSTM(Template):

    def __init__(self, input_dim, output_dim, truncate_gradient=-1, return_sequences=True,
                weight_init=OrthogonalWeight(), inner_init=GaussianWeight(mean=0, std=0.1)):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.W_i = weight_init((self.input_dim, self.output_dim))
        self.U_i = inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim), name='b_i')

        self.W_f = weight_init((self.input_dim, self.output_dim))
        self.U_f = inner_init((self.output_dim, self.output_dim))
        self.b_f = shared_ones((self.output_dim), name='b_f')

        self.W_c = weight_init((self.input_dim, self.output_dim))
        self.U_c = inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim), name='b_c')

        self.W_o = weight_init((self.input_dim, self.output_dim))
        self.U_o = inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim), name='b_o')

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]


    def _step(self, xi_t, xf_t, xo_t, xc_t,
              h_tm1, c_tm1, u_i, u_f, u_o, u_c):
        i_t = T.nnet.sigmoid(xi_t + T.dot(h_tm1, u_i))
        f_t = T.nnet.sigmoid(xf_t + T.dot(h_tm1, u_f))
        o_t = T.nnet.sigmoid(xo_t + T.dot(h_tm1, u_o))
        g_t = T.tanh(xc_t + T.dot(h_tm1, u_c))
        c_t = f_t * c_tm1 + i_t * g_t

        h_t = o_t * T.tanh(c_t)
        return h_t, c_t


    def _train_fprop(self, state_below):
        X = state_below.dimshuffle((1, 0, 2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]


class BiLSTM(Template):
    '''
    Bidirection LSTM
    '''

    def __init__(self, input_dim, output_dim, weight_init=OrthogonalWeight(),
                 inner_init=GaussianWeight(mean=0, std=0.1), truncate_gradient=-1,
                 output_mode='concat', return_sequences=False, return_idx=-1):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.output_mode = output_mode # output_mode is either sum or concatenate
        self.return_sequences = return_sequences
        self.return_idx = return_idx
        # forward weights
        self.W_i = weight_init((self.input_dim, self.output_dim))
        self.U_i = inner_init((self.output_dim, self.output_dim))
        self.b_i = shared_zeros((self.output_dim), name='b_i')

        self.W_f = weight_init((self.input_dim, self.output_dim))
        self.U_f = inner_init((self.output_dim, self.output_dim))
        self.b_f = shared_ones((self.output_dim), name='b_f')

        self.W_c = weight_init((self.input_dim, self.output_dim))
        self.U_c = inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim), name='b_c')

        self.W_o = weight_init((self.input_dim, self.output_dim))
        self.U_o = inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim), name='b_o')

        # backward weights
        self.Wb_i = weight_init((self.input_dim, self.output_dim))
        self.Ub_i = inner_init((self.output_dim, self.output_dim))
        self.bb_i = shared_zeros((self.output_dim), name='bb_i')

        self.Wb_f = weight_init((self.input_dim, self.output_dim))
        self.Ub_f = inner_init((self.output_dim, self.output_dim))
        self.bb_f = shared_ones((self.output_dim), name='bb_f')

        self.Wb_c = weight_init((self.input_dim, self.output_dim))
        self.Ub_c = inner_init((self.output_dim, self.output_dim))
        self.bb_c = shared_zeros((self.output_dim), name='bb_c')

        self.Wb_o = weight_init((self.input_dim, self.output_dim))
        self.Ub_o = inner_init((self.output_dim, self.output_dim))
        self.bb_o = shared_zeros((self.output_dim), name='bb_o')

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,

            self.Wb_i, self.Ub_i, self.bb_i,
            self.Wb_c, self.Ub_c, self.bb_c,
            self.Wb_f, self.Ub_f, self.bb_f,
            self.Wb_o, self.Ub_o, self.bb_o,
        ]


    def _forward_step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):
        i_t = T.nnet.sigmoid(xi_t + T.dot(h_tm1, u_i))
        f_t = T.nnet.sigmoid(xf_t + T.dot(h_tm1, u_f))
        o_t = T.nnet.sigmoid(xo_t + T.dot(h_tm1, u_o))
        g_t = T.tanh(xc_t + T.dot(h_tm1, u_c))
        c_t = f_t * c_tm1 + i_t * g_t
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t


    def get_forward_output(self, state_below):
        X = state_below.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_i) + self.b_i
        xf = T.dot(X, self.W_f) + self.b_f
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._forward_step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient
        )
        return outputs.dimshuffle((1,0,2))


    def get_backward_output(self, state_below):
        X = state_below.dimshuffle((1,0,2))

        xi = T.dot(X, self.Wb_i) + self.bb_i
        xf = T.dot(X, self.Wb_f) + self.bb_f
        xc = T.dot(X, self.Wb_c) + self.bb_c
        xo = T.dot(X, self.Wb_o) + self.bb_o

        [outputs, memories], updates = theano.scan(
            self._forward_step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
            ],
            non_sequences=[self.Ub_i, self.Ub_f, self.Ub_o, self.Ub_c],
            go_backwards = True,
            truncate_gradient=self.truncate_gradient
        )
        return outputs.dimshuffle((1,0,2))


    def _train_fprop(self, state_below):
        forward = self.get_forward_output(state_below)
        backward = self.get_backward_output(state_below)
        if self.output_mode == 'sum':
            output = forward + backward
        elif self.output_mode == 'concat':
            output = T.concatenate([forward, backward], axis=2)
        else:
            raise Exception('output mode is not sum or concat')
        if self.return_sequences==False:
            return output[:,self.return_idx,:]
        elif self.return_sequences==True:
            return output
        else:
            raise Exception('Unexpected output shape for return_sequences')
