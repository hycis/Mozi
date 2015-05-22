import os

import theano
import theano.tensor as T
import numpy as np

from pynet.model import AutoEncoder
from pynet.layer import *
from pynet.datasets.mnist import Mnist
from pynet.learning_rule import LearningRule
from pynet.learning_method import *
from pynet.log import Log
from pynet.train_object import TrainObject
from pynet.cost import Cost
from pynet.datasets.preprocessor import Standardize, GCN

# set environment
NNdir = os.path.dirname(os.path.realpath(__file__))
NNdir = os.path.dirname(NNdir)
NNdir = os.path.dirname(NNdir)

if not os.getenv('PYNET_DATA_PATH'):
    os.environ['PYNET_DATA_PATH'] = NNdir + '/data'

if not os.getenv('PYNET_DATABASE_PATH'):
    os.environ['PYNET_DATABASE_PATH'] = NNdir + '/database'
    if not os.path.exists(os.environ['PYNET_DATABASE_PATH']):
        os.mkdir(os.environ['PYNET_DATABASE_PATH'])

if not os.getenv('PYNET_SAVE_PATH'):
    os.environ['PYNET_SAVE_PATH'] = NNdir + '/save'
    if not os.path.exists(os.environ['PYNET_SAVE_PATH']):
        os.mkdir(os.environ['PYNET_SAVE_PATH'])

def autoencoder():

    # building dataset, batch_size and preprocessor
    data = Mnist(train_valid_test_ratio=[8,1,1], batch_size=100, preprocessor=GCN())

    # for AutoEncoder, the inputs and outputs must be the same
    train = data.get_train()
    data.set_train(train.X, train.X)

    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)

    test = data.get_test()
    data.set_test(test.X, test.X)

    # building autoencoder
    ae = AutoEncoder(input_dim = data.feature_size(), rand_seed=123)
    h1_layer = Tanh(dim=500, name='h1_layer', W=None, b=None)

    # adding encoding layer
    ae.add_encode_layer(h1_layer)

    # mirror layer has W = h1_layer.W.T
    h1_mirror = Tanh(dim=ae.input_dim, name='h1_mirror', W=h1_layer.W.T, b=None)

    # adding decoding mirror layer
    ae.add_decode_layer(h1_mirror)

    # build learning method
    learning_method = AdaGrad(learning_rate=0.1, momentum=0.9)

    # set the learning rules
    learning_rule = LearningRule(max_col_norm = 10,
                                L1_lambda = None,
                                L2_lambda = None,
                                training_cost = Cost(type='mse'),
                                learning_rate_decay_factor = None,
                                stopping_criteria = {'max_epoch' : 300,
                                                      'epoch_look_back' : 10,
                                                      'cost' : Cost(type='error'),
                                                      'percent_decrease' : 0.01}
                                )


    # put all the components into a TrainObject
    train_object = TrainObject(model = ae,
                                dataset = data,
                                learning_rule = learning_rule,
                                learning_method = learning_method)

    # finally run the training
    train_object.run()


def stacked_autoencoder():

    name = 'Stacked_AE'

    #=====[ Train First layer of stack autoencoder ]=====#
    print('Start training First Layer of AutoEncoder')


    # building dataset, batch_size and preprocessor
    data = Mnist(train_valid_test_ratio=[8,1,1], batch_size=100)

    # for AutoEncoder, the inputs and outputs must be the same
    train = data.get_train()
    data.set_train(train.X, train.X)

    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)

    test = data.get_test()
    data.set_test(test.X, test.X)

    # building autoencoder
    ae = AutoEncoder(input_dim = data.feature_size(), rand_seed=123)
    h1_layer = RELU(dim=500, name='h1_layer', W=None, b=None)

    # adding encoding layer
    ae.add_encode_layer(h1_layer)

    # mirror layer has W = h1_layer.W.T
    h1_mirror = RELU(dim=ae.input_dim, name='h1_mirror', W=h1_layer.W.T, b=None)

    # adding decoding mirror layer
    ae.add_decode_layer(h1_mirror)

    # build learning method
    learning_method = SGD(learning_rate=0.001, momentum=0.9)

    # set the learning rules
    learning_rule = LearningRule(max_col_norm = 10,
                                L1_lambda = None,
                                L2_lambda = None,
                                training_cost = Cost(type='mse'),
                                learning_rate_decay_factor = None,
                                stopping_criteria = {'max_epoch' : 3,
                                                      'epoch_look_back' : 1,
                                                      'cost' : Cost(type='error'),
                                                      'percent_decrease' : 0.01}
                                )


    # put all the components into a TrainObject
    train_object = TrainObject(model = ae,
                                dataset = data,
                                learning_rule = learning_rule,
                                learning_method = learning_method)

    # finally run the training
    train_object.run()

    #=====[ Train Second Layer of autoencoder ]=====#

    print('Start training Second Layer of AutoEncoder')


    # fprop == forward propagation
    reduced_train_X = ae.encode(train.X)
    reduced_valid_X = ae.encode(valid.X)
    reduced_test_X = ae.encode(test.X)

    data.set_train(X=reduced_train_X, y=reduced_train_X)
    data.set_valid(X=reduced_valid_X, y=reduced_valid_X)
    data.set_test(X=reduced_test_X, y=reduced_test_X)

    # create a new mlp taking inputs from the encoded outputs of first autoencoder
    ae2 = AutoEncoder(input_dim = data.feature_size(), rand_seed=None)


    h2_layer = RELU(dim=100, name='h2_layer', W=None, b=None)
    ae2.add_encode_layer(h2_layer)

    h2_mirror = RELU(dim=h1_layer.dim, name='h2_mirror', W=h2_layer.W.T, b=None)
    ae2.add_decode_layer(h2_mirror)


    train_object = TrainObject(model = ae2,
                                dataset = data,
                                learning_rule = learning_rule,
                                learning_method = learning_method)

    train_object.run()

    #=====[ Fine Tuning ]=====#
    print('Fine Tuning')

    data = Mnist()

    train = data.get_train()
    data.set_train(train.X, train.X)

    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)

    test = data.get_test()
    data.set_test(test.X, test.X)

    ae3 = AutoEncoder(input_dim = data.feature_size(), rand_seed=None)
    ae3.add_encode_layer(h1_layer)
    ae3.add_encode_layer(h2_layer)
    ae3.add_decode_layer(h2_mirror)
    ae3.add_decode_layer(h1_mirror)

    train_object = TrainObject(model = ae3,
                                dataset = data,
                                learning_rule = learning_rule,
                                learning_method = learning_method)

    train_object.run()
    print('Training Done')

if __name__ == '__main__':
    # autoencoder()
    stacked_autoencoder()
