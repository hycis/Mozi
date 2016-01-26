
from mozi.layers.template import Template
from mozi.utils.theano_utils import sharedX
from mozi.weight_init import UniformWeight

class Embedding(Template):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @output_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, init=UniformWeight(scale=0.1), weights=None):

        self.input_dim = input_dim
        self.output_dim = output_dim
        if weights is None:
            self.W = init((input_dim, output_dim))
        else:
            self.W = sharedX(weights)
        self.params = [self.W]

    def _train_fprop(self, state_below):
        state_below = state_below.astype('int32')
        return self.W[state_below]

    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)
