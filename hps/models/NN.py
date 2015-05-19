

from jobman import DD, expand, flatten


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
import pynet.layer_noise as layer_noises

import cPickle
import os

from hps.models.model import NeuralNet

import theano
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
floatX = theano.config.floatX


class NN(NeuralNet):

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
        # log.info("fine tuning")
        # for layer in train_obj.model.layers:
        #     layer.dropout_below = None
        #     layer.noise = None
        # train_obj.setup()
        # train_obj.run()







    def build_model(self, dataset):
        model = MLP(input_dim=dataset.feature_size(), rand_seed=self.state.model.rand_seed)

        hidden1 = self.build_layer(self.state.hidden1)
        # hidden2 = self.build_layer(self.state.hidden2)
        output = self.build_layer(self.state.output)
        model.add_layer(hidden1)
        # model.add_layer(hidden2)
        model.add_layer(output)
        return model
