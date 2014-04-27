import theano.tensor as T
from smartNN.cost import Cost

class LearningRule(object):
    def __init__(self, max_col_norm = 1,
                    learning_rate = 0.1,
                    momentum = 0.01,
                    momentum_type = 'normal',
                    L1_lambda = None,
                    L2_lambda = None,
                    cost = Cost(type='nll'),
                    dropout_below = 1,
                    stopping_criteria = {'max_epoch' : 100,
                                        'cost' : Cost(type='error'), 
                                        'epoch_look_back' : None, 
                                        'percent_decrease' : None}):
                                        
        self.max_col_norm = max_col_norm
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum_type = momentum_type
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.cost = cost
        self.stopping_criteria = stopping_criteria
        assert self.stopping_criteria['max_epoch'] is not None, 'max_epoch cannot be None'
        
        


