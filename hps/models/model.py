
from pynet.datasets.mnist import Mnist, Mnist_Blocks
import pynet.layer as layer
from pynet.model import *
import pynet.datasets.spec as spec
import pynet.datasets.mnist as mnist
import pynet.datasets.i2r as i2r
import pynet.datasets.recsys as recsys
import pynet.datasets.preprocessor as preproc
import pynet.learning_method as learning_methods
from pynet.learning_rule import LearningRule
from pynet.log import Log
from pynet.cost import Cost
import cPickle
import os
import pynet.datasets.dataset_noise as data_noise
import pynet.layer_noise as layer_noises

import theano
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
floatX = theano.config.floatX


class Model(object):

    def __init__(self, state):
        self.state = state


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


    def build_noise(self, noise):

        noise_obj = None if noise.type is None else \
              getattr(layer_noises, noise.type)()

        if noise.type in ['BlackOut', 'MaskOut', 'BatchOut']:
            noise_obj.ratio = noise.ratio

        elif noise.type == 'Gaussian':
            noise_obj.std = noise.std
            noise_obj.mean = noise.mean

        return noise_obj


    def build_layer(self, layer_name):
        output_noise = self.build_noise(layer_name.layer_noise)
        output = getattr(layer, layer_name.type)(dim=layer_name.dim,
                                                name=layer_name.name,
                                                dropout_below=layer_name.dropout_below,
                                                noise=output_noise)
        return output

    def build_learning_method(self):

        if self.state.learning_method.type == 'SGD':
            learn_method = getattr(learning_methods,
                           self.state.learning_method.type)(
                           learning_rate = self.state.learning_method.learning_rate,
                           momentum = self.state.learning_method.momentum)

        elif self.state.learning_method.type == 'AdaGrad':
            learn_method = getattr(learning_methods,
                           self.state.learning_method.type)(
                           learning_rate = self.state.learning_method.learning_rate,
                           momentum = self.state.learning_method.momentum)

        elif self.state.learning_method.type == 'AdaDelta':
            learn_method = getattr(learning_methods,
                           self.state.learning_method.type)(
                           rho = self.state.learning_method.rho,
                           eps = self.state.learning_method.eps)

        else:
            raise TypeError("not SGD, AdaGrad or AdaDelta")


        return learn_method


    def build_learning_rule(self):
        learning_rule = LearningRule(max_col_norm = self.state.learning_rule.max_col_norm,
                                    L1_lambda = self.state.learning_rule.L1_lambda,
                                    L2_lambda = self.state.learning_rule.L2_lambda,
                                    training_cost = Cost(type = self.state.learning_rule.cost),
                                    stopping_criteria = {'max_epoch' : self.state.learning_rule.stopping_criteria.max_epoch,
                                                        'epoch_look_back' : self.state.learning_rule.stopping_criteria.epoch_look_back,
                                                        'cost' : Cost(type=self.state.learning_rule.stopping_criteria.cost),
                                                        'percent_decrease' : self.state.learning_rule.stopping_criteria.percent_decrease})
        return learning_rule


    def build_database(self, dataset, learning_rule, learning_method, model):
        save_to_database = {'name' : self.state.log.save_to_database_name,
                            'records' : {'Dataset'          : dataset.__class__.__name__,
                                         'max_col_norm'     : learning_rule.max_col_norm,
                                         'Weight_Init_Seed' : model.rand_seed,
                                         'Dropout_Below'    : str([layer.dropout_below for layer in model.layers]),
                                         'Learning_Method'  : learning_method.__class__.__name__,
                                         'Batch_Size'       : dataset.batch_size,
                                         'Dataset_Noise'    : dataset.noise.__class__.__name__,
                                        #  'Dataset_Dir'      : dataset.data_dir,
                                         'Feature_Size'     : dataset.feature_size(),
                                         'nblocks'          : dataset.nblocks(),
                                         'Layer_Types'      : str([layer.__class__.__name__ for layer in model.layers]),
                                         'Layer_Dim'        : str([layer.dim for layer in model.layers]),
                                         'Preprocessor'     : dataset.preprocessor.__class__.__name__,
                                         'Training_Cost'    : learning_rule.cost.type,
                                         'Stopping_Cost'    : learning_rule.stopping_criteria['cost'].type}
                            }

        if learning_method.__class__.__name__ == "SGD":
            save_to_database["records"]["Learning_rate"] = learning_method.learning_rate
            save_to_database["records"]["Momentum"]    = learning_method.momentum
        elif learning_method.__class__.__name__ == "AdaGrad":
            save_to_database["records"]["Learning_rate"] = learning_method.learning_rate
            save_to_database["records"]["Momentum"]    = learning_method.momentum
        elif learning_method.__class__.__name__ == "AdaDelta":
            save_to_database["records"]["rho"] = float(learning_method.rho.get_value())
            save_to_database["records"]["eps"] = float(learning_method.eps.get_value())
        else:
            raise TypeError("not SGD, AdaGrad or AdaDelta")

        layer_noise = []
        layer_noise_params = []
        for layer in model.layers:
            layer_noise.append(layer.noise.__class__.__name__)
            if layer.noise.__class__.__name__ in ['BlackOut', 'MaskOut', 'BatchOut']:
                layer_noise_params.append(layer.noise.ratio)

            elif layer.noise.__class__.__name__ is 'Gaussian':
                layer_noise_params.append((layer.noise.mean, layer.noise.std))

            else:
                layer_noise_params.append(None)

        save_to_database["records"]["Layer_Noise"] = str(layer_noise)
        save_to_database["records"]["Layer_Noise_Params"] = str(layer_noise_params)
        save_to_database["records"]["Preprocessor_Params"] = ""

        if dataset.preprocessor.__class__.__name__ == 'Scale':
            save_to_database["records"]["Preprocessor_Params"] = str({'buffer': dataset.preprocessor.buffer,
                                                                    'max': dataset.preprocessor.max,
                                                                    'min': dataset.preprocessor.min,
                                                                    'scale_range': dataset.preprocessor.scale_range})


        return save_to_database


class NeuralNet(Model):
    def build_dataset(self):
        dataset = None

        preprocessor = None if self.state.dataset.preprocessor.type is None else \
                       getattr(preproc, self.state.dataset.preprocessor.type)()

        if self.state.dataset.preprocessor.type == 'Scale':
            preprocessor.max = self.state.dataset.preprocessor.global_max
            preprocessor.min = self.state.dataset.preprocessor.global_min
            preprocessor.buffer = self.state.dataset.preprocessor.buffer
            preprocessor.scale_range = self.state.dataset.preprocessor.scale_range

        if self.state.dataset.type == 'Mnist':
            dataset = Mnist(train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)

        if self.state.dataset.type[:11] == 'TransFactor':
            dataset = getattr(tf, self.state.dataset.type)(
                            # feature_size = self.state.dataset.feature_size,
                            # target_size = self.state.dataset.target_size,
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
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

        elif self.state.dataset.type[:20] == 'I2R_Posterior_Blocks':
            dataset = getattr(i2r, self.state.dataset.type)(
                            feature_size = self.state.dataset.feature_size,
                            target_size = self.state.dataset.feature_size,
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            one_hot = self.state.dataset.one_hot,
                            num_blocks = self.state.dataset.num_blocks,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)

        elif self.state.dataset.type[:13] == 'I2R_Posterior':
            dataset = getattr(i2r, self.state.dataset.type)(
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)


        elif self.state.dataset.type[:6] == 'RecSys':
            dataset = getattr(recsys, self.state.dataset.type)(
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)

        return dataset

class AE(Model):

    def build_dataset(self):
        dataset = None

        preprocessor = None if self.state.dataset.preprocessor.type is None else \
                       getattr(preproc, self.state.dataset.preprocessor.type)()

        noise = None if self.state.dataset.dataset_noise.type is None else \
                getattr(data_noise, self.state.dataset.dataset_noise.type)()

        if self.state.dataset.dataset_noise.type == 'Gaussian':
            noise.std = self.state.dataset.dataset_noise.std

        if self.state.dataset.preprocessor.type == 'Scale':
            preprocessor.max = self.state.dataset.preprocessor.global_max
            preprocessor.min = self.state.dataset.preprocessor.global_min
            preprocessor.buffer = self.state.dataset.preprocessor.buffer
            preprocessor.scale_range = self.state.dataset.preprocessor.scale_range

        if self.state.dataset.type == 'Mnist':
            dataset = Mnist(train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            noise = noise,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)
            train = dataset.get_train()
            dataset.set_train(train.y, train.y)
            valid = dataset.get_valid()
            dataset.set_valid(valid.y, valid.y)
            test = dataset.get_test()
            dataset.set_test(test.y, test.y)

        elif self.state.dataset.type[:12] == 'Mnist_Blocks':
            dataset = getattr(mnist, self.state.dataset.type)(
                            feature_size = self.state.dataset.feature_size,
                            target_size = self.state.dataset.feature_size,
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            noise = noise,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)

        elif self.state.dataset.type[:4] == 'P276':
            dataset = getattr(spec, self.state.dataset.type)(
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            noise = noise,
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
                            num_blocks = self.state.dataset.num_blocks,
                            preprocessor = preprocessor,
                            noise = noise,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)

        elif self.state.dataset.type[:18] == 'TransFactor_Blocks':
            dataset = getattr(tf, self.state.dataset.type)(
                            feature_size = self.state.dataset.feature_size,
                            target_size = self.state.dataset.feature_size,
                            one_hot = self.state.dataset.one_hot,
                            num_blocks = self.state.dataset.num_blocks,
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            noise = noise,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)

        elif self.state.dataset.type[:11] == 'TransFactor':
            dataset = getattr(tf, self.state.dataset.type)(
                            # feature_size = self.state.dataset.feature_size,
                            # target_size = self.state.dataset.feature_size,
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            noise = noise,
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

        elif self.state.dataset.type[:13] == 'I2R_Posterior':
            dataset = getattr(i2r, self.state.dataset.type)(
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            noise = noise,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)


        return dataset





    def build_one_hid_model(self, input_dim):
        model = AutoEncoder(input_dim=input_dim, rand_seed=self.state.model.rand_seed)

        h1_noise = self.build_noise(self.state.hidden1.layer_noise)

        hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim,
                                                        name=self.state.hidden1.name,
                                                        dropout_below=self.state.hidden1.dropout_below,
                                                        noise=h1_noise)
        model.add_encode_layer(hidden1)
        h1_mirror = getattr(layer, self.state.h1_mirror.type)(dim=input_dim,
                                                            name=self.state.h1_mirror.name,
                                                            W=hidden1.W.T,
                                                            dropout_below=self.state.h1_mirror.dropout_below)
        model.add_decode_layer(h1_mirror)
        return model



    def build_two_hid_model(self, input_dim):
        model = AutoEncoder(input_dim=input_dim, rand_seed=self.state.model.rand_seed)
        h1_noise = self.build_noise(self.state.hidden1.layer_noise)
        hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim,
                                                        name=self.state.hidden1.name,
                                                        dropout_below=self.state.hidden1.dropout_below,
                                                        noise=h1_noise)
        model.add_encode_layer(hidden1)

        h2_noise = self.build_noise(self.state.hidden2.layer_noise)
        hidden2 = getattr(layer, self.state.hidden2.type)(dim=self.state.hidden2.dim,
                                                        name=self.state.hidden2.name,
                                                        dropout_below=self.state.hidden2.dropout_below,
                                                        noise=h2_noise)
        model.add_encode_layer(hidden2)

        hidden2_mirror = getattr(layer, self.state.h2_mirror.type)(dim=self.state.hidden1.dim,
                                                                name=self.state.h2_mirror.name,
                                                                dropout_below=self.state.h2_mirror.dropout_below,
                                                                W = hidden2.W.T)
        model.add_decode_layer(hidden2_mirror)

        hidden1_mirror = getattr(layer, self.state.h1_mirror.type)(dim=input_dim,
                                                                name=self.state.h1_mirror.name,
                                                                dropout_below=self.state.h1_mirror.dropout_below,
                                                                W = hidden1.W.T)
        model.add_decode_layer(hidden1_mirror)
        return model


    def build_three_hid_model(self, input_dim):
        model = AutoEncoder(input_dim=input_dim, rand_seed=self.state.model.rand_seed)

        hidden1 = self.build_layer(self.state.hidden1)
        hidden2 = self.build_layer(self.state.hidden2)
        hidden3 = self.build_layer(self.state.hidden3)
        model.add_encode_layer(hidden1)
        model.add_encode_layer(hidden2)
        model.add_encode_layer(hidden3)

        h3_mirror = getattr(layer, self.state.h3_mirror.type)(dim=hidden2.dim,
                                                                name=self.state.h3_mirror.name,
                                                                dropout_below=self.state.h3_mirror.dropout_below,
                                                                W = hidden3.W.T)

        h2_mirror = getattr(layer, self.state.h2_mirror.type)(dim=hidden1.dim,
                                                                name=self.state.h2_mirror.name,
                                                                dropout_below=self.state.h2_mirror.dropout_below,
                                                                W = hidden2.W.T)

        h1_mirror = getattr(layer, self.state.h1_mirror.type)(dim=input_dim,
                                                                name=self.state.h1_mirror.name,
                                                                dropout_below=self.state.h1_mirror.dropout_below,
                                                                W = hidden1.W.T)


        model.add_decode_layer(h3_mirror)
        model.add_decode_layer(h2_mirror)
        model.add_decode_layer(h1_mirror)
        return model
