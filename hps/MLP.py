
from jobman import DD, expand, flatten

import pynet.layer as layer
from pynet.model import *
from pynet.layer import *
from pynet.datasets.mnist import Mnist, Mnist_Blocks
import pynet.datasets.spec as spec
import pynet.datasets.mnist as mnist
import pynet.datasets.mapping as mapping
from pynet.learning_rule import LearningRule
from pynet.log import Log
from pynet.train_object import TrainObject
from pynet.cost import Cost
import pynet.datasets.preprocessor as preproc

import cPickle
import os

import theano
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
floatX = theano.config.floatX

class NN():

    def __init__(self, state):
        self.state = state

    def run(self):
        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()
        model = self.build_model(dataset)
        database = self.build_database(dataset, learning_rule, model)
        log = self.build_log(database)
        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                model = model)
        train_obj.run()


    def build_log(self, save_to_database=None, id=None):
        log = Log(experiment_name = id is not None and '%s_%s'%(self.state.log.experiment_name,id) \
                                    or self.state.log.experiment_name,
                description = self.state.log.description,
                save_outputs = self.state.log.save_outputs,
                save_learning_rule = self.state.log.save_learning_rule,
                save_model = self.state.log.save_model,
                save_epoch_error = self.state.log.save_epoch_error,
                save_to_database = save_to_database)
        return log


    def build_dataset(self):
        dataset = None

        preprocessor = None if self.state.dataset.preprocessor is None else \
                       getattr(preproc, self.state.dataset.preprocessor)()

        if self.state.dataset.type == 'Mnist':
            dataset = Mnist(train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)


        elif self.state.dataset.type[:12] == 'Mnist_Blocks':
            dataset = getattr(mnist, self.state.dataset.type)(
                            feature_size = self.state.dataset.feature_size,
                            target_size = self.state.dataset.feature_size,
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)

        return dataset


    def build_learning_rule(self):
        learning_rule = LearningRule(max_col_norm = self.state.learning_rule.max_col_norm,
                                    learning_rate = self.state.learning_rule.learning_rate,
                                    momentum = self.state.learning_rule.momentum,
                                    momentum_type = self.state.learning_rule.momentum_type,
                                    L1_lambda = self.state.learning_rule.L1_lambda,
                                    L2_lambda = self.state.learning_rule.L2_lambda,
                                    training_cost = Cost(type = self.state.learning_rule.cost),
                                    stopping_criteria = {'max_epoch' : self.state.learning_rule.stopping_criteria.max_epoch,
                                                        'epoch_look_back' : self.state.learning_rule.stopping_criteria.epoch_look_back,
                                                        'cost' : Cost(type=self.state.learning_rule.stopping_criteria.cost),
                                                        'percent_decrease' : self.state.learning_rule.stopping_criteria.percent_decrease})
        return learning_rule


    def build_model(self, dataset):
        model = MLP(input_dim=dataset.feature_size(), rand_seed=self.state.model.rand_seed)

        hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim,
                                                          name=self.state.hidden1.name,
                                                          dropout_below=self.state.hidden1.dropout_below,
                                                          blackout_below=self.state.hidden1.blackout_below)
        model.add_layer(hidden1)

        output = getattr(layer, self.state.output.type)(dim=dataset.target_size(),
                                                        name=self.state.output.name,
                                                        dropout_below=self.state.output.dropout_below,
                                                        blackout_below=self.state.output.blackout_below)
        model.add_layer(output)
        return model



    def build_database(self, dataset, learning_rule, model):
        save_to_database = {'name' : self.state.log.save_to_database_name,
                            'records' : {'Dataset'          : dataset.__class__.__name__,
                                         'max_col_norm'     : learning_rule.max_col_norm,
                                         'Weight_Init_Seed' : model.rand_seed,
                                         'Dropout_Below'    : str([layer.dropout_below for layer in model.layers]),
                                         'Blackout_Below'   : str([layer.blackout_below for layer in model.layers]),
                                         'Batch_Size'       : dataset.batch_size,
                                         'nblocks'          : dataset.nblocks(),
                                         'Layer_Types'      : str([layer.__class__.__name__ for layer in model.layers]),
                                         'Layer_Dim'        : str([layer.dim for layer in model.layers]),
                                         'Preprocessor'     : dataset.preprocessor.__class__.__name__,
                                         'Learning_Rate'    : learning_rule.learning_rate,
                                         'Momentum'         : learning_rule.momentum,
                                         'Training_Cost'    : learning_rule.cost.type,
                                         'Stopping_Cost'    : learning_rule.stopping_criteria['cost'].type}
                            }
        return save_to_database
