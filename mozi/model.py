
import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

class Sequential(object):

    def __init__(self, **kwargs):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def pop(self, index):
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
        input_state = self.layers[0].input_var
        output, stats = self.test_fprop(input_state)
        f = theano.function([input_state], output)
        return f(input_values)

    def get_layers(self):
        return self.layers
