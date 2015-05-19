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

class Laura_Two_Layers(AE):

    def __init__(self, state):
        self.state = state


    def build_model(self, input_dim):
        with open(os.environ['PYNET_SAVE_PATH'] + '/'
                    + self.state.hidden1.model + '/model.pkl') as f1:
            model1 = cPickle.load(f1)

        with open(os.environ['PYNET_SAVE_PATH'] + '/'
                    + self.state.hidden2.model + '/model.pkl') as f2:
            model2 = cPickle.load(f2)

        model = AutoEncoder(input_dim=input_dim)
        while len(model1.encode_layers) > 0:
            model.add_encode_layer(model1.pop_encode_layer())
        while len(model2.encode_layers) > 0:
            model.add_encode_layer(model2.pop_encode_layer())
        while len(model2.decode_layers) > 0:
            model.add_decode_layer(model2.pop_decode_layer())
        while len(model1.decode_layers) > 0:
            model.add_decode_layer(model1.pop_decode_layer())
        return model

    def run(self):

        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()
        learn_method = self.build_learning_method()

        model = self.build_model(dataset.feature_size())
        model.layers[0].dropout_below = self.state.hidden1.dropout_below

        if self.state.log.save_to_database_name:
            database = self.build_database(dataset, learning_rule, learn_method, model)
            database['records']['h1_model'] = self.state.hidden1.model
            database['records']['h2_model'] = self.state.hidden2.model
            log = self.build_log(database)

        log.info("Fine Tuning")
        for layer in model.layers:
            layer.dropout_below = None
            layer.noise = None

        train_obj = TrainObject(log = log,
                                dataset = dataset,
                                learning_rule = learning_rule,
                                learning_method = learn_method,
                                model = model)

        train_obj.run()
