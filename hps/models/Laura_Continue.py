
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


class Laura_Continue(AE):

    def __init__(self, state):
        self.state = state


    def build_model(self):
        with open(os.environ['PYNET_SAVE_PATH'] + '/'
                    + self.state.hidden1.model + '/model.pkl') as f:
            model = cPickle.load(f)
        return model

    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()
        learn_method = self.build_learning_method()

        model = self.build_model()

        if self.state.fine_tuning_only:
            for layer in model.layers:
                layer.dropout_below = None
                layer.noise = None
            print "Fine Tuning Only"

        if self.state.log.save_to_database_name:
            database = self.build_database(dataset, learning_rule, learn_method, model)
            database['records']['model'] = self.state.hidden1.model
            log = self.build_log(database)

        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                learning_method = learn_method,
                                model = model)

        train_obj.run()

        if not self.state.fine_tuning_only:
            log.info("..Fine Tuning after Noisy Training")
            for layer in train_obj.model.layers:
                layer.dropout_below = None
                layer.noise = None
            train_obj.setup()
            train_obj.run()
