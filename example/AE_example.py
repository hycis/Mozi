import os

import theano
import theano.tensor as T
import numpy as np

from pynet.model import MLP, AutoEncoder
from pynet.layer import RELU, Sigmoid, Softmax, Linear
from pynet.datasets.mnist import Mnist
from pynet.datasets.spec import P276
from pynet.learning_rule import LearningRule
from pynet.log import Log
from pynet.train_object import TrainObject
from pynet.cost import Cost
from pynet.datasets.preprocessor import Standardize, GCN

NNdir = os.path.dirname(os.path.realpath(__file__))
NNdir = os.path.dirname(NNdir)

if not os.getenv('PYNET_DATA_PATH'):
    os.environ['PYNET_DATA_PATH'] = os.path.dirname(NNdir) + '/smartNN/data'

if not os.getenv('PYNET_DATABASE_PATH'):
    os.environ['PYNET_DATABASE_PATH'] = NNdir + '/database'

if not os.getenv('PYNET_SAVE_PATH'):
    os.environ['PYNET_SAVE_PATH'] = NNdir + '/save'

print('PYNET_DATA_PATH = ' + os.environ['PYNET_DATA_PATH'])
print('PYNET_SAVE_PATH = ' + os.environ['PYNET_SAVE_PATH'])
print('PYNET_DATABASE_PATH = ' + os.environ['PYNET_DATABASE_PATH'])


def autoencoder():



    learning_rule = LearningRule(max_col_norm = None,
                            learning_rate = 0.01,
                            momentum = 0.1,
                            momentum_type = 'normal',
                            L1_lambda = None,
                            L2_lambda = None,
                            training_cost = Cost(type='mse'),
                            stopping_criteria = {'max_epoch' : 100,
                                                'cost' : Cost(type='mse'),
                                                'epoch_look_back' : 10,
                                                'percent_decrease' : 0.001}
                            )

    # building dataset
    data = Mnist(train_valid_test_ratio=[5,1,1])

    train = data.get_train()
    data.set_train(train.X, train.X)

    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)

    test = data.get_test()
    data.set_test(test.X, test.X)

    # building autoencoder
    ae = AutoEncoder(input_dim = data.feature_size(), rand_seed=None)
    h1_layer = RELU(dim=100, name='h1_layer', W=None, b=None)

    # adding encoding layer
    ae.add_encode_layer(h1_layer)

    # adding decoding mirror layer
    ae.add_decode_layer(Sigmoid(dim=data.target_size(), name='output_layer', W=h1_layer.W.T, b=None))

    log = Log(experiment_name = 'AE',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            save_to_database = {'name': 'Database.db',
                                'records' : {'Dataset' : data.__class__.__name__,
                                             'Weight_Init_Seed' : ae.rand_seed,
                                             'Dropout_Below' : str([layer.dropout_below for layer in ae.layers]),
                                             'Batch_Size' : data.batch_size,
                                             'Layer_Size' : len(ae.layers),
                                             'Layer_Dim' : str([layer.dim for layer in ae.layers]),
                                             'Preprocessor' : data.preprocessor.__class__.__name__,
                                             'Learning_Rate' : learning_rule.learning_rate,
                                             'Momentum' : learning_rule.momentum}}
            ) # end log

    log.info(ae.rand_seed)
    log.info('this is testing')

    train_object = TrainObject(model = ae,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)

    train_object.run()


def stacked_autoencoder():

    name = 'Stacked_AE'

    #=====[ Train First layer of stack autoencoder ]=====#
    print('Start training First Layer of AutoEncoder')


    log = Log(experiment_name = name + '_layer1',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            save_to_database = {'name': 'Database.db',
                                'records' : {'Dataset' : data.__class__.__name__,
                                             'Weight_Init_Seed' : ae.rand_seed,
                                             'Dropout_Below' : str([layer.dropout_below for layer in ae.layers]),
                                             'Batch_Size' : data.batch_size,
                                             'Layer_Size' : len(ae.layers),
                                             'Layer_Dim' : str([layer.dim for layer in ae.layers]),
                                             'Preprocessor' : data.preprocessor.__class__.__name__,
                                             'Learning_Rate' : learning_rule.learning_rate,
                                             'Momentum' : learning_rule.momentum}}
            ) # end log

    learning_rule = LearningRule(max_col_norm = None,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                L1_lambda = None,
                                L2_lambda = None,
                                cost = Cost(type='mse'),
                                stopping_criteria = {'max_epoch' : 3,
                                                    'epoch_look_back' : 1,
                                                    'cost' : Cost(type='mse'),
                                                    'percent_decrease' : 0.001}
                                )

    data = Mnist()

    train = data.get_train()
    data.set_train(train.X, train.X)

    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)

    test = data.get_test()
    data.set_test(test.X, test.X)

    ae = AutoEncoder(input_dim = data.feature_size(), rand_seed=123)

    h1_layer = RELU(dim=500, name='h1_layer', W=None, b=None)
    ae.add_encode_layer(h1_layer)
    h1_mirror = RELU(dim = data.target_size(), name='h1_mirror', W=h1_layer.W.T, b=None)
    ae.add_decode_layer(h1_mirror)


    train_object = TrainObject(model = ae,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)

    train_object.run()

    #=====[ Train Second Layer of autoencoder ]=====#

    print('Start training Second Layer of AutoEncoder')

    log2 = Log(experiment_name = name + '_layer2',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')

    learning_rule = LearningRule(max_col_norm = None,
                            learning_rate = 0.01,
                            momentum = 0.1,
                            momentum_type = 'normal',
                            L1_lambda = None,
                            L2_lambda = None,
                            cost = Cost(type='mse'),
                            stopping_criteria = {'max_epoch' : 3,
                                                'epoch_look_back' : 1,
                                                'cost' : Cost(type='mse'),
                                                'percent_decrease' : 0.001}
                            )

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
                            log = log2)

    train_object.run()

    #=====[ Fine Tuning ]=====#
    print('Fine Tuning')

    log3 = Log(experiment_name = name + '_full',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            save_to_database = {'name': 'Database.db',
                                'records' : {'Dataset' : data.__class__.__name__,
                                             'Weight_Init_Seed' : ae.rand_seed,
                                             'Dropout_Below' : str([layer.dropout_below for layer in ae.layers]),
                                             'Batch_Size' : data.batch_size,
                                             'Layer_Size' : len(ae.layers),
                                             'Layer_Dim' : str([layer.dim for layer in ae.layers]),
                                             'Preprocessor' : data.preprocessor.__class__.__name__,
                                             'Learning_Rate' : learning_rule.learning_rate,
                                             'Momentum' : learning_rule.momentum}}
            ) # end log

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
                            log = log3)

    train_object.run()
    print('Training Done')

if __name__ == '__main__':
    autoencoder()
#     stacked_autoencoder()
