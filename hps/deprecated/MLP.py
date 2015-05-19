
from jobman import DD, expand, flatten

import pynet.layer as layer
from pynet.model import *
from pynet.layer import *
from pynet.datasets.mnist import Mnist, Mnist_Blocks
import pynet.datasets.spec as spec
import pynet.datasets.mnist as mnist
import pynet.datasets.transfactor as tf
import pynet.datasets.mapping as mapping
import pynet.learning_method as learning_methods
from pynet.learning_rule import LearningRule
from pynet.log import Log
from pynet.train_object import TrainObject
from pynet.cost import Cost
import pynet.datasets.preprocessor as preproc
import pynet.datasets.dataset_noise as noisy
import pynet.layer_noise as layer_noise

import cPickle
import os

from AE import AE

import theano
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
floatX = theano.config.floatX

class NN(AE):

    def __init__(self, state):
        self.state = state

    def run(self):
        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()
        model = self.build_model(dataset)
        learn_method = self.build_learning_method()
        database = self.build_database(dataset, learning_rule, learn_method, model)
        log = self.build_log(database)
        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                learning_method = learn_method,
                                model = model)
        train_obj.run()
        log.info("fine tuning")
        for layer in train_obj.model.layers:
            layer.dropout_below = None
            layer.noise = None
        train_obj.setup()
        train_obj.run()


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

        return dataset


    def build_layer(self, dataset, layer_name):

        output_noise = None if layer_name.layer_noise.type is None else \
              getattr(layer_noise, layer_name.layer_noise.type)()
        if layer_name.layer_noise.type in ['BlackOut', 'MaskOut', 'BatchOut']:
            output_noise.ratio = layer_name.layer_noise.ratio

        elif layer_name.layer_noise.type is 'Gaussian':
            output_noise.std = layer_name.layer_noise.std
            output_noise.mean = layer_name.layer_noise.mean

        output = getattr(layer, layer_name.type)(dim=layer_name.dim,
                                                        name=layer_name.name,
                                                        dropout_below=layer_name.dropout_below,
                                                        noise=output_noise)
        return output

    def build_model(self, dataset):
        model = MLP(input_dim=dataset.feature_size(), rand_seed=self.state.model.rand_seed)

        # h1_noise = None if self.state.hidden1.layer_noise.type is None else \
        #       getattr(layer_noise, self.state.hidden1.layer_noise.type)()
        # if self.state.hidden1.layer_noise.type in ['BlackOut', 'MaskOut', 'BatchOut']:
        #     h1_noise.ratio = self.state.hidden1.layer_noise.ratio
        #
        # elif self.state.hidden1.layer_noise.type is 'Gaussian':
        #     h1_noise.std = self.state.hidden1.layer_noise.std
        #     h1_noise.mean = self.state.hidden1.layer_noise.mean
        #
        # hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim,
        #                                                 name=self.state.hidden1.name,
        #                                                 dropout_below=self.state.hidden1.dropout_below,
        #                                                 noise=h1_noise)

        hidden1 = self.build_layer(dataset, self.state.hidden1)
        # hidden2 = self.build_layer(dataset, self.state.hidden2)
        output = self.build_layer(dataset, self.state.output)
        model.add_layer(hidden1)
        # model.add_layer(hidden2)

        # output_noise = None if self.state.output.layer_noise.type is None else \
        #       getattr(layer_noise, self.state.output.layer_noise.type)()
        # if self.state.output.layer_noise.type in ['BlackOut', 'MaskOut', 'BatchOut']:
        #     output_noise.ratio = self.state.output.layer_noise.ratio
        #
        # elif self.state.output.layer_noise.type is 'Gaussian':
        #     output_noise.std = self.state.output.layer_noise.std
        #     output_noise.mean = self.state.output.layer_noise.mean
        #
        # output = getattr(layer, self.state.output.type)(dim=dataset.target_size(),
        #                                                 name=self.state.output.name,
        #                                                 dropout_below=self.state.output.dropout_below,
        #                                                 noise=output_noise)

        model.add_layer(output)
        return model
