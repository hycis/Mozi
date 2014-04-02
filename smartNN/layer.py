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
        raise NotImplementedError(str(type(self))+" does not implement _test_layer_stats.")

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
        raise NotImplementedError(str(type(self))+" does not implement _train_layer_stats.")


    
class Linear(Layer):
    def _test_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output

    def _train_fprop(self, state_below):
        output = self._linear_part(state_below)
        return output   
    
 
    # This is called every batch, the final cout will be the mean of all batches in an epoch
    def _test_layer_stats(self, layer_output):
        max_norm_sqr = self.W ** 2
        max_norm_sqr = T.max(max_norm_sqr.sum(axis=1))
#         mean_norm_sqr = T.mean(max_norm_sqr.sum(axis=1))
        max_norm = T.sqrt(max_norm_sqr)
#         mean_norm = T.sqrt(mean_norm_sqr)
        return [('max_norm', max_norm),
#                 ('mean_norm', mean_norm), 
                ('max', T.max(layer_output)), 
                ('min', T.min(layer_output))]

    
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
        max_norm_sqr = self.W ** 2
        max_norm_sqr = T.max(max_norm_sqr.sum(axis=1))
#         mean_norm_sqr = T.mean(max_norm_sqr.sum(axis=1))
        max_norm = T.sqrt(max_norm_sqr)
#         mean_norm = T.sqrt(mean_norm_sqr)
        return [('max_norm', max_norm),
                ('max', T.max(layer_output)), 
                ('min', T.min(layer_output))]
    
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
        max_norm_sqr = self.W ** 2
        max_norm_sqr = T.max(max_norm_sqr.sum(axis=1))
#         mean_norm_sqr = T.mean(max_norm_sqr.sum(axis=1), axis=0)
        max_norm = T.sqrt(max_norm_sqr)
#         mean_norm = T.sqrt(mean_norm_sqr)
        return [('max_norm', max_norm),
#                 ('mean_norm', mean_norm), 
                ('max', T.max(layer_output)), 
                ('min', T.min(layer_output))]
    
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
        max_norm_sqr = self.W ** 2
        max_norm_sqr = T.max(max_norm_sqr.sum(axis=1))
#         mean_norm_sqr = T.mean(max_norm_sqr.sum(axis=1))
        max_norm = T.sqrt(max_norm_sqr)
#         mean_norm = T.sqrt(mean_norm_sqr)
        return [('max_norm', max_norm),
#                 ('mean_norm', mean_norm), 
                ('max', T.max(layer_output)), 
                ('min', T.min(layer_output))]
    
    def _train_layer_stats(self, layer_output):
        return self._test_layer_stats(layer_output)
    
                        
