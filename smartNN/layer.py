import numpy as np
import theano
import theano.tensor as T

class Layer(object):
    """
    Abstract Class
    """

    def __init__(self, dim, name, W=None, b=None):        
        self.dim = dim
        self.name = name
        
        self.W = W
        self.b = b
        
        if self.W is not None and self.W.name is None:
            self.W.name = 'W_' + self.name
        if self.b is not None and self.b.name is None:    
            self.b.name = 'b_' + self.name

    def _test_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _test_fprop.")

    def _train_fprop(self, state_below):
        raise NotImplementedError(str(type(self))+" does not implement _train_fprop.")
    
    def _linear_part(self, state_below):
        return T.dot(state_below, self.W) + self.b
    
    def _test_layer_stats(self, layer_output):
        """
        DESCRIPTION:
            This method is called every batch, the final result will be the mean of all the
            results from all the batches in an epoch from the test set.
        PARAM:
            layer_output: the output from the layer
        RETURN:
            A list of tuples of [('name_a', var_a), ('name_b', var_b)] whereby dim(var_b)=0 
        """
#         length_sqr_sum = (self.W ** 2).sum(axis=1)
#     
#         max_length_sqr = T.max(length_sqr_sum)
#         max_length = T.sqrt(max_length_sqr)
        
        w_len = T.sqrt((self.W ** 2).sum(axis=1))
        max_length = T.max(w_len)
        mean_length = T.mean(w_len)
        min_length = T.min(w_len)


#         mean_length_sqr = T.mean(length_sqr_sum)
#         mean_length = T.sqrt(mean_length_sqr)
#         
#         min_length_sqr = T.min(length_sqr_sum)
#         min_length = T.sqrt(min_length_sqr)
        
        return [('max_length', max_length),
                ('mean_length', mean_length),
                ('min_length', min_length), 
                ('max', T.max(layer_output)), 
                ('min', T.min(layer_output))]
                
    def _train_layer_stats(self, layer_output):
        """
        DESCRIPTION:
            This method is called every batch, the final result will be the mean of all the
            results from all the batches in an epoch from the train set.
        PARAM:
            layer_output: the output from the layer
        RETURN:
            A list of tuples of [('name_a', var_a), ('name_b', var_b)] whereby dim(var_b)=0 
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
    
                        
