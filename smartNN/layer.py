import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Layer(object):
    """
    Abstract Class
    """

    def __init__(self, dim, name, W=None, b=None, dropout=None):  
        """
        DESCRIPTION:
            This is an abstract layer class
        PARAM:
            dim(int): dimension of the layer
            name(string): name of the layer
            W(tensor variable): Weight of 2D tensor matrix
            b(tensor variable): bias of 2D tensor matrix
            dropout(int): probability of the inputs from the layer below been masked out
        """    
         
        self.dim = dim
        self.name = name
        self.W = W
        self.b = b
        self.dropout = dropout
        self.theano_rand = RandomStreams()
        
        if self.W is not None and self.W.name is None:
            self.W.name = 'W_' + self.name
        if self.b is not None and self.b.name is None:    
            self.b.name = 'b_' + self.name

    def _test_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _test_fprop.")

    def _train_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _train_fprop.")
    
    def _linear_part(self, state_below):
        """
            DESCRIPTION:
                performs linear transform y = dot(W, state_below) + b
            PARAM:
                state_below: 1d array of inputs from layer below
        """
    
        if self.dropout is not None:
            assert self.dropout >= 0 and self.dropout <= 1, 'dropout is not in range [0,1]'
            state_below = self.theano_rand.binomial(size=(self.dim,), 
                                                n=1, p=(1-self.dropout),
                                                dtype=theano.config.floatX) * state_below
        
        return T.dot(state_below, self.W) + self.b
    
    def _test_layer_stats(self, layer_output):
        """
        DESCRIPTION:
            This method is called every batch, the final result will be the mean of all the
            results from all the batches in an epoch from the test set.
        PARAM:
            layer_output: the output from the layer
        RETURN:
            A list of tuples of [('name_a', var_a), ('name_b', var_b)] whereby var is scalar 
        """
        
        w_len = T.sqrt((self.W ** 2).sum(axis=1))
        max_length = T.max(w_len)
        mean_length = T.mean(w_len)
        min_length = T.min(w_len)
        
        return [('max_col_length', max_length),
                ('mean_col_length', mean_length),
                ('min_col_length', min_length), 
                ('output_max', T.max(layer_output)),
                ('output_mean', T.mean(layer_output)), 
                ('output_min', T.min(layer_output)),
                ('max_W', T.max(self.W)),
                ('min_W', T.min(self.W)),
                ('mean_W', T.mean(self.W)),
                ('max_b', T.max(self.b)),
                ('min_b', T.min(self.b)),
                ('mean_b', T.mean(self.b))]
                
    def _train_layer_stats(self, layer_output):
        """
        DESCRIPTION:
            This method is called every batch, the final result will be the mean of all the
            results from all the batches in an epoch from the train set.
        PARAM:
            layer_output: the output from the layer
        RETURN:
            A list of tuples of [('name_a', var_a), ('name_b', var_b)] whereby var is scalar 
        """
        return self._test_layer_stats(layer_output)


    
class Linear(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output

    def _train_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output   
    
 
    # This is called every batch, the final cout will be the mean of all batches in an epoch
    def _test_layer_stats(self, layer_output):
        return super(Linear, self)._test_layer_stats(layer_output)

    
    def _train_layer_stats(self, layer_output):
        return self._test_layer_stats(layer_output)

class Sigmoid(Layer):
        
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.nnet.sigmoid(output)

    def _train_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.nnet.sigmoid(output)   
    
 
    # This is called every batch, the final cout will be the mean of all batches in an epoch
    def _test_layer_stats(self, layer_output):
        return super(Sigmoid, self)._test_layer_stats(layer_output)
    
    def _train_layer_stats(self, layer_output):
        return self._test_layer_stats(layer_output)
    

class RELU(Layer):

    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output * (output > 0.)

    def _train_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output * (output > 0.)   
    
 
    # This is called every batch, the final cout will be the mean of all batches in an epoch
    def _test_layer_stats(self, layer_output):
        return super(RELU, self)._test_layer_stats(layer_output)
    
    def _train_layer_stats(self, layer_output):
        return self._test_layer_stats(layer_output)


  
class Softmax(Layer):
       
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.nnet.softmax(output)

    def _train_fprop(self, state_below):
        output = self._linear_part(state_below)
        return T.nnet.softmax(output)   
    
 
    # This is called every batch, the final cout will be the mean of all batches in an epoch
    def _test_layer_stats(self, layer_output):
        return super(Softmax, self)._test_layer_stats(layer_output)
    
    def _train_layer_stats(self, layer_output):
        return self._test_layer_stats(layer_output)
    
                        
