import theano
import theano.tensor as T
floatX = theano.config.floatX

import numpy as np

import time, datetime
import sys

import logging
internal_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from pynet.log import Log
from pynet.utils.utils import split_list, generate_shared_list, \
                                merge_lists, get_shared_values, \
                                duplicate_param

from pynet.utils.check_memory import print_mem_usage


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

    def __init__(self, model, dataset, learning_rule, log=None):
        self.model = model
        self.dataset = dataset
        self.learning_rule = learning_rule
        self.log = log

        if self.log is None:
            # use default Log setting
            self.log = Log(logger=internal_logger)

        elif self.log.save_to_database:
            self.log.print_records()

        self.log.info( '..begin setting up train object')
        self.setup()

    def setup(self):

        #================[ check output dim with target size ]================#

        assert self.model.layers[-1].dim == self.dataset.target_size(), \
                'output dim: ' + str(self.model.layers[-1].dim) + \
                ', is not equal to target size: ' + str(self.dataset.target_size())


        assert self.model.input_dim == self.dataset.feature_size(), \
                'input dim: ' + str(self.model.input_dim) + \
                ', is not equal to feature size: ' + str(self.dataset.feature_size())

        #===================[ build params and deltas list ]==================#


        def is_shared_var(var):
            return var.__class__.__name__ == 'TensorSharedVariable' or \
                    var.__class__.__name__ == 'CudaNdarraySharedVariable'


        params = []
        deltas = []

        prev_layer_dim = self.model.input_dim
        for layer in self.model.layers:

            if is_shared_var(layer.W):
                assert layer.W.dtype == floatX
                params += [layer.W]
                deltas += [theano.shared(np.zeros((prev_layer_dim, layer.dim), dtype=floatX))]

            else:
                self.log.info(layer.W.name + ' is ' + layer.W.__class__.__name__ +
                            ' but not SharedVariable.')

            if is_shared_var(layer.b):
                assert layer.b.dtype == floatX
                params += [layer.b]
                deltas += [theano.shared(np.zeros(layer.dim, dtype=floatX))]

            else:
                self.log.info(layer.b.name + ' is ' + layer.b.__class__.__name__ +
                            ' but not SharedVariable.')

            prev_layer_dim = layer.dim

        #=====================[ training params updates ]=====================#

        self.log.info("..update params: " + str(params))

        train_x = T.matrix('train_x', dtype=floatX)
        train_y = T.matrix('train_y', dtype=floatX)

        assert self.learning_rule.momentum_type == 'normal' or \
                self.learning_rule.momentum_type == 'nesterov', \
                'momentum is not normal | nesterov'

        if self.learning_rule.momentum_type == 'normal':

            train_y_pred, train_layers_stats = self.model.train_fprop(train_x)
            train_cost = self.learning_rule.cost.get_cost(train_y, train_y_pred)

            if self.learning_rule.L1_lambda is not None:
                self.log.info('..applying L1_lambda: %f'%self.learning_rule.L1_lambda)
                L1 = theano.shared(0.)
                for layer in self.model.layers:
                    if is_shared_var(layer.W):
                        L1 += T.sqrt((layer.W ** 2).sum(axis=0)).sum()

                    else:
                        self.log.info(layer.W.name + ' is ' + layer.W.__class__.__name__ +
                            ' is not used in L1 regularization')
                train_cost += self.learning_rule.L1_lambda * L1

            if self.learning_rule.L2_lambda is not None:
                self.log.info('..applying L2_lambda: %f'%self.learning_rule.L2_lambda)
                L2 = theano.shared(0.)
                for layer in self.model.layers:
                    if is_shared_var(layer.W):
                        L2 += ((layer.W ** 2).sum(axis=0)).sum()

                    else:
                        self.log.info(layer.W.name + ' is ' + layer.W.__class__.__name__ +
                            ' is not used in L2 regularization')
                train_cost += self.learning_rule.L2_lambda * L2

            train_updates = []
            gparams = T.grad(train_cost, params)
            for delta, param, gparam in zip(deltas, params, gparams):
                train_updates += [(delta, self.learning_rule.momentum * delta
                            - self.learning_rule.learning_rate * gparam)]

                # applying max_col_norm regularisation
                if param.name[0] == 'W' and self.learning_rule.max_col_norm is not None:
                    W_update = param + delta
                    w_len = T.sqrt((W_update ** 2).sum(axis=0))
                    divisor = (w_len <= self.learning_rule.max_col_norm) + \
                            (w_len > self.learning_rule.max_col_norm) * w_len / \
                            self.learning_rule.max_col_norm
                    W_update = W_update / divisor.reshape((1, divisor.shape[0]))
                    train_updates += [(param, W_update)]

                else:
                    train_updates += [(param, param + delta)]

        # TODO
        elif self.learning_rule.momentum_type == 'nesterov':
            raise NotImplementedError('nesterov not implemented yet')

        #----[ append updates of stats from each layer to train updates ]-----#

        self.train_stats_names, train_stats_vars = split_list(train_layers_stats)
        self.train_stats_shared = generate_shared_list(train_stats_vars)
        train_stats_updates = merge_lists(self.train_stats_shared, train_stats_vars)
        train_updates += train_stats_updates

        #-------------------------[ train functions ]-------------------------#

        self.log.info('..begin compiling functions')

        train_stopping_cost = self.learning_rule.stopping_criteria['cost'].get_cost(train_y, train_y_pred)

        self.training = theano.function(inputs=[train_x, train_y],
                                        outputs=(train_stopping_cost, train_cost),
                                        updates=train_updates,
                                        on_unused_input='warn',
                                        allow_input_downcast=True)

        self.log.info('..training function compiled')

        #======================[ testing params updates ]=====================#

        test_x = T.matrix('test_x', dtype=floatX)
        test_y = T.matrix('test_y', dtype=floatX)
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
                                        on_unused_input='warn',
                                        allow_input_downcast=True)

        self.log.info('..testing function compiled')


    def run(self):

        best_train_error = float("inf")
        best_valid_error = float("inf")
        best_test_error = float("inf")

        mean_train_error = float("inf")
        mean_valid_error = float("inf")
        mean_test_error = float("inf")

        mean_train_cost = float("inf")
        mean_valid_cost = float("inf")
        mean_test_cost = float("inf")

        train_stats_values = []
        valid_stats_values = []
        test_stats_values = []

        epoch = 1
        best_epoch = 1
        error_dcr = 0
        self.best_epoch_so_far = 0

        train_stats_names = ['train_' + name for name in self.train_stats_names]
        valid_stats_names = ['valid_' + name for name in self.test_stats_names]
        test_stats_names = ['test_' + name for name in self.test_stats_names]

        job_start = time.time()

        while (self.continue_learning(epoch, error_dcr, best_valid_error)):

            start_time = time.time()

            num_train_examples = 0
            total_train_cost = 0.
            total_train_stopping_cost = 0.
            train_stats_values = np.zeros(len(train_stats_names), dtype=floatX)

            num_valid_examples = 0
            total_valid_cost = 0.
            total_valid_stopping_cost = 0.
            valid_stats_values = np.zeros(len(valid_stats_names), dtype=floatX)

            num_test_examples = 0
            total_test_cost = 0.
            total_test_stopping_cost = 0.
            test_stats_values = np.zeros(len(test_stats_names), dtype=floatX)

            blk = 0

            for block in self.dataset:
                block_time = time.time()
                blk += 1

                train_set = block.get_train()
                valid_set = block.get_valid()
                test_set = block.get_test()


                #====================[ Training Progress ]====================#
                if train_set.dataset_size() > 0:

                    self.log.info('..training '+ self.dataset.__class__.__name__
                                + ' block %s/%s'%(blk, self.dataset.nblocks()))

                    for idx in train_set:
                        stopping_cost, cost = self.training(train_set.X[idx], train_set.y[idx])
                        total_train_cost += cost * len(idx)
                        total_train_stopping_cost += stopping_cost * len(idx)
                        num_train_examples += len(idx)
                        train_stats_values += len(idx) * get_shared_values(self.train_stats_shared)


                #===================[ Validating Progress ]===================#
                if valid_set.dataset_size() > 0:

                    self.log.info('..validating ' + self.dataset.__class__.__name__
                                + ' block %s/%s'%(blk, self.dataset.nblocks()))

                    for idx in valid_set:
                        stopping_cost, cost = self.testing(valid_set.X[idx], valid_set.y[idx])
                        total_valid_cost += cost * len(idx)
                        total_valid_stopping_cost += stopping_cost * len(idx)
                        num_valid_examples += len(idx)
                        valid_stats_values += len(idx) * get_shared_values(self.test_stats_shared)


                #====================[ Testing Progress ]=====================#
                if test_set.dataset_size() > 0:

                    self.log.info('..testing ' + self.dataset.__class__.__name__
                                + ' block %s/%s'%(blk, self.dataset.nblocks()))

                    for idx in test_set:
                        stopping_cost, cost = self.testing(test_set.X[idx], test_set.y[idx])
                        total_test_cost += cost * len(idx)
                        total_test_stopping_cost += stopping_cost * len(idx)
                        num_test_examples += len(idx)
                        test_stats_values += len(idx) * get_shared_values(self.test_stats_shared)

                self.log.info('block time: %0.2fs'%(time.time()-block_time))
                self.log.info(print_mem_usage())

            #==============[ Update best cost and error values ]==============#
            if train_set.dataset_size() > 0:
                mean_train_error = total_train_stopping_cost / num_train_examples
                mean_train_cost = total_train_cost / num_train_examples
                train_stats_values /= num_train_examples

                if mean_train_error < best_train_error:
                    best_train_error = mean_train_error

            if valid_set.dataset_size() > 0:
                mean_valid_error = total_valid_stopping_cost / num_valid_examples
                mean_valid_cost = total_valid_cost / num_valid_examples
                valid_stats_values /= num_valid_examples

                if best_valid_error - mean_valid_error > 0:
                    error_dcr = best_valid_error - mean_valid_error
                    best_valid_error = mean_valid_error

            if test_set.dataset_size() > 0:
                mean_test_error = total_test_stopping_cost / num_test_examples
                mean_test_cost = total_test_cost / num_test_examples
                test_stats_values /= num_test_examples

            #=======[ save model, save learning_rule, save to database ]======#
            if mean_test_error < best_test_error:

                best_test_error = mean_test_error
                best_epoch = epoch

                if self.log.save_model:
                    self.log._save_model(self.model)
                    self.log.info('..model saved')

                if self.log.save_learning_rule:
                    self.log._save_learning_rule(self.learning_rule)
                    self.log.info('..learning rule saved')

            #==============[ save to database, save epoch error]==============#
            if self.log.save_to_database:
                self.log._save_to_database(epoch, best_train_error, best_valid_error, best_test_error)
                self.log.info('..sent to database: %s:%s' % (self.log.save_to_database['name'],
                                                        self.log.experiment_name))

            if self.log.save_epoch_error:
                self.log._save_epoch_error(best_epoch, best_train_error, best_valid_error, best_test_error)
                self.log.info('..epoch error saved')

            end_time = time.time()

            #=====================[ log outputs to file ]=====================#

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
            self.log._log_outputs(outputs)
            epoch += 1

        job_end = time.time()
        self.log.info('Job Completed on %s'%time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(job_end)))
        ttl_time = int(job_end - job_start)
        dt = datetime.timedelta(seconds=ttl_time)
        self.log.info('Total Time Taken: %s\n'%str(dt))


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
