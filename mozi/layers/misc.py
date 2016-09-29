
import theano
import theano.tensor as T
from mozi.layers.template import Template
import numpy as np

class Flatten(Template):

    def _train_fprop(self, state_below):
        size = T.prod(state_below.shape) / state_below.shape[0]
        nshape = (state_below.shape[0], size)
        return T.reshape(state_below, nshape)


class Reshape(Template):

    def __init__(self, dims):
        self.params = []
        self.dims = dims

    def _train_fprop(self, state_below):
        nshape = (state_below.shape[0],) + self.dims
        return T.reshape(state_below, nshape)


class Transform(Template):

    def __init__(self, dims):
        '''
        Reshaping the data such that the first dim alters when the rest of the
        dim is altered. If X of shape (a, b, c, d) and input dims of shape (d, e),
        then return shape will be (a*b*c*d/(d*e), d, e). Useful for tranforming
        data in RNN/LSTM with mlp layers before recurrent layers.
        '''
        self.params = []
        self.dims = dims

    def _train_fprop(self, state_below):
        first_dim = T.prod(state_below.shape) / np.prod(self.dims)
        return T.reshape(state_below, (first_dim,)+self.dims)


class Crop(Template):
    def __init__(self, border):
        self.border = border
        self.params = []
        assert len(self.border) == 2

    def _train_fprop(self, state_below):
        w, h = self.border
        return state_below[:,:,h:-h,w:-w]


class Parallel(Template):

    def __init__(self, *models):
        self.models = models
        self.params = []
        for model in self.models:
            for layer in model.layers:
                self.params += layer.params

    def _train_fprop(self, state_below):
        rstate = []
        for model, state in zip(self.models, state_below):
            out, _ = model.train_fprop(state)
            rstate.append(out)
        return rstate
