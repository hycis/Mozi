__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"


import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

class Model(object):
    '''
    Abstract Class
    '''

    def __init__(self, input_dim, rand_seed=None):
        self.input_dim = input_dim
        self.layers = []

        if rand_seed is None:
            self.rand_seed = np.random.randint(1, 10000)
        else:
            self.rand_seed = rand_seed
        np.random.seed(self.rand_seed)

    def test_fprop(self, input_state):
        raise NotImplementedError(str(type(self))+" does not implement test_fprop.")

    def train_fprop(self, input_state):
        raise NotImplementedError(str(type(self))+" does not implement train_fprop.")



class MLP(Model):

    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def add_layer(self, layer):

        prev_layer_dim = 0

        if len(self.layers) == 0:
            prev_layer_dim = self.input_dim
        else:
            prev_layer_dim = self.layers[-1].dim

        if layer.W is None:
            W_values = np.asarray(np.random.uniform(
                                low = -4 * np.sqrt(6. / (layer.dim + prev_layer_dim)),
                                high = 4 * np.sqrt(6. / (layer.dim + prev_layer_dim)),
                                size = (prev_layer_dim, layer.dim)),
                                dtype = floatX)

            # W_values = np.zeros(shape=(prev_layer_dim, layer.dim), dtype=floatX)

            layer.W = theano.shared(value=W_values, name='W_'+layer.name, borrow=True)

        if layer.b is None:
            layer.b = theano.shared(np.zeros(layer.dim, dtype=floatX),
                                    name='b_'+layer.name, borrow=True)

        self.layers.append(layer)

    def pop_layer(self, index):
        return self.layers.pop(index)

    def test_fprop(self, input_state):
        test_layers_stats = []
        for i in xrange(len(self.layers)):
            layer_output = self.layers[i]._test_fprop(input_state)
            stats = self.layers[i]._layer_stats(input_state, layer_output)
            input_state = layer_output
            class_name = self.layers[i].__class__.__name__
            stats = [(str(i)+'_'+class_name+'_'+a, b) for (a,b) in stats]
            test_layers_stats += stats

        return input_state, test_layers_stats


    def train_fprop(self, input_state):
        train_layers_stats = []
        for i in xrange(len(self.layers)):
            layer_output = self.layers[i]._train_fprop(input_state)
            stats = self.layers[i]._layer_stats(input_state, layer_output)
            input_state = layer_output
            class_name = self.layers[i].__class__.__name__
            stats = [(str(i)+'_'+class_name+'_'+a, b) for (a,b) in stats]
            train_layers_stats += stats

        return input_state, train_layers_stats

    def fprop(self, input_values):
        input_state = T.matrix('X', dtype=floatX)
        output, stats = self.test_fprop(input_state)
        f = theano.function([input_state], output)
        return f(input_values)

    def get_layers(self):
        return self.layers


class AutoEncoder(MLP):

    def __init__(self, **kwargs):

        self.encode_layers = []
        self.decode_layers = []

        super(AutoEncoder, self).__init__(**kwargs)


    def add_encode_layer(self, layer):
        self.add_layer(layer)
        self.encode_layers.append(layer)

    def add_decode_layer(self, layer):
        self.add_layer(layer)
        self.decode_layers.append(layer)

    def pop_encode_layer(self, index=0):
        layer = self.encode_layers.pop(index)
        self.layers.remove(layer)
        return layer

    def pop_decode_layer(self, index=0):
        layer = self.decode_layers.pop(index)
        self.layers.remove(layer)
        return layer

    def _fprop(self, layers, input_state):
        for layer in layers:
            input_state = layer._test_fprop(input_state)
        return input_state

    def encode(self, input_values):
        input_state = T.matrix('X', dtype=floatX)
        output_state = self._fprop(self.encode_layers, input_state)
        f = theano.function([input_state], output_state)
        return f(input_values)

    def decode(self, input_values):
        input_state = T.matrix('X', dtype=floatX)
        output_state = self._fprop(self.decode_layers, input_state)
        f = theano.function([input_state], output_state)
        return f(input_values)
