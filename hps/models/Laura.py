

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

from hps.models.model import AE

import theano
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
floatX = theano.config.floatX

class Laura(AE):

    def __init__(self, state):
        self.state = state


    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()
        learn_method = self.build_learning_method()

        if self.state.num_layers == 1:
            model = self.build_one_hid_model(dataset.feature_size())
        elif self.state.num_layers == 2:
            model = self.build_two_hid_model(dataset.feature_size())
        elif self.state.num_layers == 3:
            model = self.build_three_hid_model(dataset.feature_size())
        else:
            raise ValueError()

        database = self.build_database(dataset, learning_rule, learn_method, model)
        log = self.build_log(database)

        dataset.log = log

        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                learning_method = learn_method,
                                model = model)

        train_obj.run()

        log.info("Fine Tuning")

        for layer in train_obj.model.layers:
            layer.dropout_below = None
            layer.noise = None

        train_obj.setup()
        train_obj.run()
