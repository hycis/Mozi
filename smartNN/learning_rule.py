import theano.tensor as T
from smartNN.cost import Cost

class LearningRule(object):
    def __init__(self, max_norm = 1,
                    learning_rate = 0.1,
                    momentum = 0.01,
                    momentum_type = 'normal',
                    weight_decay = 0,
                    cost = Cost(type='nll'),
                    dropout = 1,
                    stopping_criteria = {'max_epoch' : 100, 
                                        'epoch_look_back' : None, 
                                        'accu_increase' : None}):
                                        
        self.max_norm = max_norm
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_type = momentum_type
        self.weight_decay = weight_decay
        self.cost = cost
        self.dropout = dropout
        self.stopping_criteria = stopping_criteria
        assert self.stopping_criteria['max_epoch'] is not None, 'max_epoch cannot be None'
        
        


