
from jobman import DD, expand, flatten

import pynet.layer as layer
from pynet.model import *
from pynet.layer import *
from pynet.datasets.mnist import Mnist, Mnist_Blocks
import pynet.datasets.spec as spec
import pynet.datasets.mnist as mnist
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

class AE:

    def __init__(self, state):
        self.state = state

    def run(self):
        log = self.build_log()
        dataset = self.build_dataset()

        learning_rule = self.build_learning_rule()
        model = self.build_model(dataset)
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
                save_hyperparams = self.state.log.save_hyperparams,
                save_model = self.state.log.save_model,
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
            train = dataset.get_train()
            dataset.set_train(train.X, train.X)
            valid = dataset.get_valid()
            dataset.set_valid(valid.X, valid.X)
            test = dataset.get_test()
            dataset.set_test(test.X, test.X)

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

        elif self.state.dataset.type[:4] == 'P276':
            dataset = getattr(spec, self.state.dataset.type)(
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)
            train = dataset.get_train()
            dataset.set_train(train.X, train.X)
            valid = dataset.get_valid()
            dataset.set_valid(valid.X, valid.X)
            test = dataset.get_test()
            dataset.set_test(test.X, test.X)

        elif self.state.dataset.type[:5] == 'Laura':
            dataset = getattr(spec, self.state.dataset.type)(
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


    def build_one_hid_model(self, input_dim):
        model = AutoEncoder(input_dim = input_dim, rand_seed=self.state.model.rand_seed)
        hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim,
                                                        name=self.state.hidden1.name,
                                                        dropout_below=self.state.hidden1.dropout_below)
        model.add_encode_layer(hidden1)
        h1_mirror = getattr(layer, self.state.h1_mirror.type)(dim=input_dim,
                                                            name=self.state.h1_mirror.name,
                                                            W=hidden1.W.T,
                                                            dropout_below=self.state.h1_mirror.dropout_below)
        model.add_decode_layer(h1_mirror)
        return model

    def build_one_hid_model_no_transpose(self, input_dim):
        model = AutoEncoder(input_dim = input_dim, rand_seed=self.state.model.rand_seed)
        hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim,
                                                        name=self.state.hidden1.name,
                                                        dropout_below=self.state.hidden1.dropout_below)
        model.add_encode_layer(hidden1)
        h1_mirror = getattr(layer, self.state.h1_mirror.type)(dim=input_dim,
                                                            name=self.state.h1_mirror.name,
                                                            dropout_below=self.state.h1_mirror.dropout_below)
        model.add_decode_layer(h1_mirror)
        return model

    def build_two_hid_model(self, input_dim):
        model = AutoEncoder(input_dim=input_dim, rand_seed=self.state.model.rand_seed)
        hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim,
                                                        name=self.state.hidden1.name,
                                                        dropout_below=self.state.hidden1.dropout_below)
        model.add_encode_layer(hidden1)

        hidden2 = getattr(layer, self.state.hidden2.type)(dim=self.state.hidden2.dim,
                                                        name=self.state.hidden2.name,
                                                        dropout_below=self.state.hidden2.dropout_below)
        model.add_encode_layer(hidden2)

        hidden2_mirror = getattr(layer, self.state.h2_mirror.type)(dim=self.state.hidden1.dim,
                                                                name=self.state.h2_mirror.name,
                                                                W = hidden2.W.T)
        model.add_decode_layer(hidden2_mirror)

        hidden1_mirror = getattr(layer, self.state.h1_mirror.type)(dim=input_dim,
                                                                name=self.state.h1_mirror.name,
                                                                W = hidden1.W.T)
        model.add_decode_layer(hidden1_mirror)
        return model


    def build_database(self, dataset, learning_rule, model):
        save_to_database = {'name' : self.state.log.save_to_database_name,
                            'records' : {'Dataset'          : dataset.__class__.__name__,
                                         'max_col_norm'     : learning_rule.max_col_norm,
                                         'Weight_Init_Seed' : model.rand_seed,
                                         'Dropout_Below'    : str([layer.dropout_below for layer in model.layers]),
                                         'Batch_Size'       : dataset.batch_size,
                                         'nblocks'          : dataset.nblocks(),
                                         'Layer_Size'       : len(model.layers),
                                         'Layer_Types'      : str([layer.__class__.__name__ for layer in model.layers]),
                                         'Feature_Size'     : dataset.feature_size(),
                                         'Layer_Dim'        : str([layer.dim for layer in model.layers]),
                                         'Preprocessor'     : dataset.preprocessor.__class__.__name__,
                                         'Learning_Rate'    : learning_rule.learning_rate,
                                         'Momentum'         : learning_rule.momentum,
                                         'Training_Cost'    : learning_rule.cost.type,
                                         'Stopping_Cost'    : learning_rule.stopping_criteria['cost'].type}
                            }

        return save_to_database

class AE_Testing(AE):

    def __init__(self, state):
        self.state = state


    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()
        model = self.build_one_hid_model(dataset.feature_size())

        if self.state.log.save_to_database_name:
            database = self.build_database(dataset, learning_rule, model)
            log = self.build_log(database)

        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                model = model)
        train_obj.run()

        # fine tuning
        log.info("fine tuning")
        train_obj.model.layers[0].dropout_below = None
        train_obj.setup()
        train_obj.run()


class Laura(AE):

    def __init__(self, state):
        self.state = state


    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()

        if self.state.num_layers == 1:
            model = self.build_one_hid_model(dataset.feature_size())
        elif self.state.num_layers == 2:
            model = self.build_two_hid_model(dataset.feature_size())
        else:
            raise ValueError()

        database = self.build_database(dataset, learning_rule, model)
        log = self.build_log(database)


        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                model = model)

        train_obj.run()

        # fine tuning
        log.info("fine tuning")
        train_obj.model.layers[0].dropout_below = None
        train_obj.setup()
        train_obj.run()

class Laura_Two_Layers(AE):

    def __init__(self, state):
        self.state = state


    def build_model(self, input_dim):
        with open(os.environ['PYNET_SAVE_PATH'] + '/log/'
                    + self.state.hidden1.model + '/model.pkl') as f1:
            model1 = cPickle.load(f1)

        with open(os.environ['PYNET_SAVE_PATH'] + '/log/'
                    + self.state.hidden2.model + '/model.pkl') as f2:
            model2 = cPickle.load(f2)

        model = AutoEncoder(input_dim=input_dim)
        model.add_encode_layer(model1.pop_encode_layer())
        model.add_encode_layer(model2.pop_encode_layer())
        model.add_decode_layer(model2.pop_decode_layer())
        model.add_decode_layer(model1.pop_decode_layer())
        return model

    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()

        model = self.build_model(dataset.feature_size())
        model.layers[0].dropout_below = self.state.hidden1.dropout_below

        if self.state.log.save_to_database_name:
            database = self.build_database(dataset, learning_rule, model)
            database['records']['h1_model'] = self.state.hidden1.model
            database['records']['h2_model'] = self.state.hidden2.model
            log = self.build_log(database)

        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                model = model)

        log.info("fine tuning")
        train_obj.model.layers[0].dropout_below = None
        train_obj.model.layers[1].dropout_below = None
        train_obj.run()

class Laura_Three_Layers(AE):
    def __init__(self, state):
        self.state = state


    def build_model(self, input_dim):
        with open(os.environ['PYNET_SAVE_PATH'] + '/log/'
                    + self.state.hidden1.model + '/model.pkl') as f1:
            model1 = cPickle.load(f1)

        with open(os.environ['PYNET_SAVE_PATH'] + '/log/'
                    + self.state.hidden2.model + '/model.pkl') as f2:
            model2 = cPickle.load(f2)

        with open(os.environ['PYNET_SAVE_PATH'] + '/log/'
                    + self.state.hidden3.model + '/model.pkl') as f3:
            model3 = cPickle.load(f3)

        model = AutoEncoder(input_dim=input_dim)

        model.add_encode_layer(model1.pop_encode_layer())
        model.add_encode_layer(model2.pop_encode_layer())
        model.add_encode_layer(model3.pop_encode_layer())
        model.add_decode_layer(model3.pop_decode_layer())
        model.add_decode_layer(model2.pop_decode_layer())
        model.add_decode_layer(model1.pop_decode_layer())

        return model

    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()

        model = self.build_model(dataset.feature_size())
        model.layers[0].dropout_below = self.state.hidden1.dropout_below

        if self.state.log.save_to_database_name:
            database = self.build_database(dataset, learning_rule, model)
            database['records']['h1_model'] = self.state.hidden1.model
            database['records']['h2_model'] = self.state.hidden2.model
            database['records']['h3_model'] = self.state.hidden3.model
            log = self.build_log(database)

        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                model = model)

        # train_obj.run()

        # fine tuning
        log.info("fine tuning")
        # train_obj.model.layers[0].dropout_below = None
        # train_obj.setup()
        train_obj.run()


class Laura_Two_Layers_No_Transpose(AE):

    def __init__(self, state):
        self.state = state


    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()

        if self.state.num_layers == 1:
            model = self.build_one_hid_model_no_transpose(dataset.feature_size())
        else:
            raise ValueError()

        if self.state.log.save_to_database_name:
            database = self.build_database(dataset, learning_rule, model)
            log = self.build_log(database)

        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                model = model)

        train_obj.run()

        # fine tuning
        log.info("fine tuning")
        train_obj.model.layers[0].dropout_below = None
        train_obj.setup()
        train_obj.run()
