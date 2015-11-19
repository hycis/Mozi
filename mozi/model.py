
import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

class Sequential(object):

    def __init__(self, input_var, output_var):
        """
        PARAM:
            input_var (T.vector() | T.matrix() | T.tensor3() | T.tensor4()):
                    The tensor variable input to the model that corresponds to
                    the number of dimensions of the input X of dataset
            input_var (T.vector() | T.matrix() | T.tensor3() | T.tensor4()):
                    The tensor variable output from the model that corresponds to
                    the number of dimensions of the output y of dataset

        """
        self.input_var = input_var
        self.output_var = output_var
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def pop(self, index):
        return self.layers.pop(index)

    def test_fprop(self, input_state, layers=None):
        test_layers_stats = []
        if layers is None:
            layers = xrange(len(self.layers))
        for i in layers:
            layer_output = self.layers[i]._test_fprop(input_state)
            stats = self.layers[i]._layer_stats(input_state, layer_output)
            input_state = layer_output
            class_name = self.layers[i].__class__.__name__
            stats = [(str(i)+'_'+class_name+'_'+a, b) for (a,b) in stats]
            test_layers_stats += stats

        return input_state, test_layers_stats


    def train_fprop(self, input_state, layers=None):
        train_layers_stats = []
        if layers is None:
            layers = xrange(len(self.layers))
        for i in layers:
            layer_output = self.layers[i]._train_fprop(input_state)
            stats = self.layers[i]._layer_stats(input_state, layer_output)
            input_state = layer_output
            class_name = self.layers[i].__class__.__name__
            stats = [(str(i)+'_'+class_name+'_'+a, b) for (a,b) in stats]
            train_layers_stats += stats

        return input_state, train_layers_stats


    def fprop(self, input_values):
        return self.fprop_layers(input_values)


    def fprop_layers(self, input_values, layers=None):
        output, stats = self.test_fprop(self.input_var, layers)
        f = theano.function([self.input_var], output, on_unused_input='warn', allow_input_downcast=True)
        return f(input_values)


    def get_layers(self):
        return self.layers
