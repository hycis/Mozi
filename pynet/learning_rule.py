
from pynet.cost import Cost

class LearningRule(object):
    def __init__(self, max_col_norm = 1,
                    L1_lambda = None,
                    L2_lambda = None,
                    training_cost = Cost(type='nll'),
                    learning_rate_decay_factor = None,
                    stopping_criteria = {'max_epoch' : 100,
                                        'cost' : Cost(type='error'),
                                        'epoch_look_back' : None,
                                        'percent_decrease' : None}):

        self.max_col_norm = max_col_norm
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.cost = training_cost
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.stopping_criteria = stopping_criteria
        assert self.stopping_criteria['max_epoch'] is not None, 'max_epoch cannot be None'
