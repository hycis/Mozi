import theano
import theano.tensor as T

import numpy as np

import time
import sys
import logging
log = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

from smartNN.utils.utils import split_list, generate_shared_list, \
                                merge_lists, get_shared_values, \
                                duplicate_param

class TrainObject():

    '''
    UPDATES:
        (Normal momentum)
        delta := momentum * delta - learning_rate * (d cost(param) / d param)
        param := param + delta
        
        (Nesterov momentum)
        delta := momentum * delta - learning_rate * (d cost(param + momentum*delta) / d param)
        param := param + delta
    '''

    def __init__(self, model, dataset, learning_rule, log):
        self.model = model
        self.dataset = dataset
        self.learning_rule = learning_rule
        self.log = log
    
        self._setup()
    
    def _setup(self):
        
        log.info('..begin setting up train object')
        
        #===================[ build params and deltas list ]==================#
        
        params = []
        deltas = []
        
        prev_layer_dim = self.model.input_dim
        for layer in self.model.layers:
            if layer.W.__class__.__name__ == 'TensorSharedVariable' or \
                layer.W.__class__.__name__ == 'CudaNdarraySharedVariable':
                params += [layer.W]
                deltas += [theano.shared(np.zeros((prev_layer_dim, layer.dim), 
                                        dtype=theano.config.floatX))]
            
            else:            
                log.warning(layer.W.name + ' is ' + layer.W.__class__.__name__ + 
                            ' but not TensorSharedVariable.')

            if layer.b.__class__.__name__ == 'TensorSharedVariable' or \
                layer.b.__class__.__name__ == 'CudaNdarraySharedVariable':
                params += [layer.b]
                deltas += [theano.shared(np.zeros(layer.dim, dtype=theano.config.floatX))]
            
            else:            
                log.warning(layer.b.name + ' is ' + layer.b.__class__.__name__ + 
                            ' but not TensorSharedVariable.')
            
            prev_layer_dim = layer.dim
        
        #=====================[ training params updates ]=====================#            
        
        log.info("..number of update params ", len(params))
        
        train_x = T.matrix('train_x')
        train_y = T.matrix('train_y')
        
        assert self.learning_rule.momentum_type == 'normal' or \
                self.learning_rule.momentum_type == 'nesterov', \
                'momentum is not normal | nesterov'
        
        train_updates = []
        
        if self.learning_rule.momentum_type == 'normal':

            train_y_pred, train_layers_stats = self.model.train_fprop(train_x)            
            train_cost = self.learning_rule.cost.get_cost(train_y, train_y_pred)
            gparams = T.grad(train_cost, params)
            
            for delta, param, gparam in zip(deltas, params, gparams):
                train_updates += [(delta, self.learning_rule.momentum * delta 
                            - self.learning_rule.learning_rate * gparam)]
                
                # applying max_col_norm regularisation
                if param.name[0] == 'W' and self.learning_rule.max_col_norm is not None:
                    W_update = param + delta
                    w_len = T.sqrt((W_update ** 2).sum(axis=1))
                    divisor = (w_len <= self.learning_rule.max_col_norm) + \
                            (w_len > self.learning_rule.max_col_norm) * w_len / \
                            self.learning_rule.max_col_norm
                    W_update = W_update / divisor.reshape(((divisor.shape[0]),1))
                    train_updates += [(param, W_update)]
                
                else:
                    train_updates += [(param, param + delta)]
            
        elif self.learning_rule.momentum_type == 'nesterov':
            raise NotImplementedError('nesterov not implemented yet')
        
        #----[ append updates of stats from each layer to train updates ]-----#
        
        self.train_stats_names, train_stats_vars = split_list(train_layers_stats)
        self.train_stats_shared = generate_shared_list(train_stats_vars)
        train_stats_updates = merge_lists(self.train_stats_shared, train_stats_vars)
        train_updates += train_stats_updates 
        
        #-------------------------[ train functions ]-------------------------#
        
        log.info('..begin compiling functions')

        train_stopping_cost = self.learning_rule.stopping_criteria['cost'].get_cost(train_y, train_y_pred)
        
        self.training = theano.function(inputs=[train_x, train_y], 
                                        outputs=(train_stopping_cost, train_cost), 
                                        updates=train_updates,
                                        on_unused_input='warn')
        
        log.info('..training function compiled')
        
        #======================[ testing params updates ]=====================#

        test_x = T.matrix('test_x')
        test_y = T.matrix('test_y')
        test_y_pred, test_layers_stats = self.model.test_fprop(test_x)
        
        #-----[ append updates of stats from each layer to test updates ]-----#

        self.test_stats_names, test_stats_vars = split_list(test_layers_stats)
        self.test_stats_shared = generate_shared_list(test_stats_vars)
        test_stats_updates = merge_lists(self.test_stats_shared, test_stats_vars)
                
        #-------------------------[ test functions ]--------------------------#

        test_stopping_cost = self.learning_rule.stopping_criteria['cost'].get_cost(test_y, test_y_pred)
        test_cost = self.learning_rule.cost.get_cost(test_y, test_y_pred)
                                          
        self.testing = theano.function(inputs=[test_x, test_y], 
                                        outputs=(test_stopping_cost, test_cost),
                                        updates=test_stats_updates,
                                        on_unused_input='warn')
        
        log.info('..testing function compiled')
        
                
    def run(self):
    
        train_set = self.dataset.get_train()
        valid_set = self.dataset.get_valid()
        test_set = self.dataset.get_test()
        
        best_train_error = float("inf")
        best_valid_error = float("inf")
        best_test_error = float("inf")
        
        mean_train_error = float("inf")
        mean_valid_error = float("inf")
        mean_test_error = float("inf")
        
        mean_train_cost = float("inf")
        mean_valid_cost = float("inf")
        mean_test_cost = float("inf")
        
        train_stats_names = []
        train_stats_values = []
        
        valid_stats_names = []
        valid_stats_values = []
        
        test_stats_names = []
        test_stats_values = []
        
        epoch = 1
        error_dcr = 0
        self.best_epoch_so_far = 0               
        
        while (self.continue_learning(epoch, error_dcr, best_valid_error)):

            start_time = time.time()
            
            #======================[ Training Progress ]======================#
            if train_set is not None:
                
                log.info('..training in progress')

                assert train_set.feature_size() == self.model.input_dim and \
                        train_set.target_size() == self.model.layers[-1].dim, \
                        'train_set input or target size does not match the model ' + \
                        'input or target size. ' + \
                        '\ntrain_set feature size: ' + str(train_set.feature_size()) + \
                        '\nmodel input dim: ' + str(self.model.input_dim) + \
                        '\ntrain_set target size: ' + str(train_set.target_size()) + \
                        '\nmodel output dim: ' + str(self.model.layers[-1].dim)
                
                num_examples = 0
                total_cost = 0.
                total_stopping_cost = 0. 

                train_stats_names = ['train_' + name for name in self.train_stats_names]
                train_stats_values = np.zeros(len(train_stats_names), dtype=theano.config.floatX)
                
                for idx in train_set:
                    stopping_cost, cost = self.training(train_set.X[idx], train_set.y[idx])

                    total_cost += cost * len(idx)
                    total_stopping_cost += stopping_cost * len(idx)
                    num_examples += len(idx)
                    
                    train_stats_values += len(idx) * get_shared_values(self.train_stats_shared)
                    
                mean_train_error = total_stopping_cost / num_examples
                mean_train_cost = total_cost / num_examples
                
                train_stats_values /= num_examples
                
                if mean_train_error < best_train_error:
                    best_train_error = mean_train_error
                
            #=====================[ Validating Progress ]=====================#
            if valid_set is not None:

                log.info('..validating in progress')

                assert valid_set.feature_size() == self.model.input_dim and \
                        valid_set.target_size() == self.model.layers[-1].dim, \
                        'valid_set input or target size does not match the model ' + \
                        'input or target size. ' + \
                        '\nvalid_set feature size: ' + str(valid_set.feature_size()) + \
                        '\nmodel input dim: ' + str(self.model.input_dim) + \
                        '\nvalid_set target size: ' + str(valid_set.target_size()) + \
                        '\nmodel output dim: ' + str(self.model.layers[-1].dim)
                                    
                num_examples = 0
                total_cost = 0.
                total_stopping_cost = 0. 
                
                valid_stats_names = ['valid_' + name for name in self.test_stats_names] 
                valid_stats_values = np.zeros(len(valid_stats_names), dtype=theano.config.floatX)
                
                for idx in valid_set:
                    stopping_cost, cost = self.testing(train_set.X[idx], train_set.y[idx])

                    total_cost += cost * len(idx)
                    total_stopping_cost += stopping_cost * len(idx)
                    num_examples += len(idx)
                    
                    valid_stats_values += len(idx) * get_shared_values(self.test_stats_shared)
               
                mean_valid_error = total_stopping_cost / num_examples
                mean_valid_cost = total_cost / num_examples
                
                valid_stats_values /= num_examples
                
                if best_valid_error - mean_valid_error > 0:
                    error_dcr = best_valid_error - mean_valid_error
                    best_valid_error = mean_valid_error
            
            #======================[ Testing Progress ]=======================#
            if test_set is not None:
            
                log.info('..testing in progress')

                assert test_set.feature_size() == self.model.input_dim and \
                        test_set.target_size() == self.model.layers[-1].dim, \
                        'test_set input or target size does not match the model ' + \
                        'input or target size. ' + \
                        '\ntest_set feature size: ' + str(test_set.feature_size()) + \
                        '\nmodel input dim: ' + str(self.model.input_dim) + \
                        '\ntest_set target size: ' + str(test_set.target_size()) + \
                        '\nmodel output dim: ' + str(self.model.layers[-1].dim)
                        
                num_examples = 0
                total_cost = 0.
                total_stopping_cost = 0. 
                
                test_stats_names = ['test_' + name for name in self.test_stats_names]
                test_stats_values = np.zeros(len(test_stats_names), dtype=theano.config.floatX)

                for idx in test_set:
                    stopping_cost, cost = self.testing(train_set.X[idx], train_set.y[idx])

                    total_cost += cost * len(idx)
                    total_stopping_cost += stopping_cost * len(idx)
                    num_examples += len(idx)
                    
                    test_stats_values += len(idx) * get_shared_values(self.test_stats_shared)

                test_stats_values /= num_examples
    
                mean_test_error = total_stopping_cost / num_examples
                mean_test_cost = total_cost / num_examples
            
                #======[ save model, save hyperparams, send to database ]=====#
                if mean_test_error < best_test_error:
                
                    best_test_error = mean_test_error

                    if self.log is not None and self.log.save_model:
                        self.log._save_model(self.model)
                        log.info('..model saved')
                
                    if self.log is not None and self.log.save_hyperparams:
                        self.log._save_hyperparams(self.learning_rule)
                        log.info('..hyperparams saved')

                    if self.log is not None and self.log.send_to_database:
                        self.log._send_to_database(self.learning_rule,
                                                    best_train_error,
                                                    best_valid_error,
                                                    best_test_error,
                                                    self.dataset.batch_size,
                                                    len(self.model.layers),
                                                    str([layer.dim for layer in self.model.layers]))
                                                    
                        log.info('..sent to database: %s:%s' % (self.log.send_to_database, 
                                                                self.log.experiment_id))

            
            end_time = time.time()
            
            #=====================[ log outputs to file ]=====================#
            if self.log is not None and self.log.save_outputs:

                merged_train = merge_lists(train_stats_names, train_stats_values)
                merged_valid = merge_lists(valid_stats_names, valid_stats_values)
                merged_test = merge_lists(test_stats_names, test_stats_values)
            
                stopping_cost_type = self.learning_rule.stopping_criteria['cost'].type
                outputs = [('epoch', epoch),
                            ('runtime(s)', int(end_time-start_time)),
                            ('mean_train_cost_' + self.learning_rule.cost.type, mean_train_cost),
                            ('mean_train_error_' + stopping_cost_type, mean_train_error),
                            ('best_train_error_' + stopping_cost_type, best_train_error),
                            ('mean_valid_cost_' + self.learning_rule.cost.type, mean_valid_cost),
                            ('mean_valid_error_' + stopping_cost_type, mean_valid_error),
                            ('best_valid_error_' + stopping_cost_type, best_valid_error),
                            ('mean_test_cost_' + self.learning_rule.cost.type, mean_test_cost),
                            ('mean_test_error_' + stopping_cost_type, mean_test_error),
                            ('best_test_error_' + stopping_cost_type, best_test_error)]
                            
                outputs += merged_train + merged_valid + merged_test
            
                self.log._save_outputs(outputs)

            epoch += 1
            
            
    def continue_learning(self, epoch, error_dcr, best_valid_error):

        if epoch > self.learning_rule.stopping_criteria['max_epoch']:
            return False
        
        elif self.learning_rule.stopping_criteria['percent_decrease'] is None or \
            self.learning_rule.stopping_criteria['epoch_look_back'] is None:
            return True
            
        elif np.abs(error_dcr * 1.0 / best_valid_error) \
            >= self.learning_rule.stopping_criteria['percent_decrease']:
            self.best_epoch_so_far = epoch
            return True
        
        elif epoch - self.best_epoch_so_far > \
            self.learning_rule.stopping_criteria['epoch_look_back']:
            return False
        
        else:
            return True
    
    
    
            
                
    
    


    
            
        