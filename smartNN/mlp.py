
import numpy as np
import theano
import theano.tensor as T


class MLP(object):

    def __init__(self, input_dim, rand_seed=123):
        self.input_dim = input_dim
        self.rand_seed = rand_seed
        self.layers = []
        np.random.seed(self.rand_seed)
        
        
    def add_layer(self, layer):
        
        prev_layer_dim = 0
        
        if len(self.layers) == 0:
            prev_layer_dim = self.input_dim
        else:
            prev_layer_dim = self.layers[-1].dim
            
        if layer.W is None:
            W_values = np.asarray(np.random.uniform(
                                low = -np.sqrt(1. / (prev_layer_dim)),
                                high = np.sqrt(1. / (prev_layer_dim)),
                                size = (prev_layer_dim, layer.dim)), 
                                dtype = theano.config.floatX)
            
            layer.W = theano.shared(value=W_values, name='W_'+layer.name, borrow=True)

        if layer.b is None:
            layer.b = theano.shared(np.zeros(layer.dim, dtype=theano.config.floatX),
                                    name='b_'+layer.name, borrow=True)
        
        self.layers.append(layer)
            
    def pop_layer(self, i):
        self.layers.pop(i)

    def test_fprop(self, input_state):
        
        test_layers_stats = []
        for i in xrange(len(self.layers)):            
            input_state = self.layers[i]._test_fprop(input_state)
            stats = self.layers[i]._test_layer_stats(input_state)
            class_name = self.layers[i].__class__.__name__
            stats = [(str(i)+'_'+class_name+'_'+a, b) for (a,b) in stats]
            test_layers_stats += stats
            
        return input_state, test_layers_stats
            
    
    def train_fprop(self, input_state):
    
        train_layers_stats = []
        for i in xrange(len(self.layers)):            
            input_state = self.layers[i]._train_fprop(input_state)
            stats = self.layers[i]._train_layer_stats(input_state)
            class_name = self.layers[i].__class__.__name__
            stats = [(str(i)+'_'+class_name+'_'+a, b) for (a,b) in stats]
            train_layers_stats += stats
            
        return input_state, train_layers_stats
        
    def fprop(self, input_values):
        input_state = T.matrix('X', dtype=theano.config.floatX)
        output, stats = self.test_fprop(input_state)
        f = theano.function([input_state], output)
        return f(input_values)

    def get_layers(self):
        return self.layers
            

    
    

 