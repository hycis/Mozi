import os

import theano
import theano.tensor as T
import numpy as np

from smartNN.mlp import MLP
from smartNN.layer import RELU, Sigmoid, Softmax, Linear
from smartNN.datasets.mnist import Mnist
from smartNN.datasets.spec import P276
from smartNN.learning_rule import LearningRule
from smartNN.log import Log
from smartNN.train_object import TrainObject
from smartNN.cost import Cost
from smartNN.datasets.preprocessor import Standardize, GCN

NNdir = os.path.dirname(os.path.realpath(__file__))
NNdir = os.path.dirname(NNdir)


if not os.getenv('smartNN_DATA_PATH'):
    os.environ['smartNN_DATA_PATH'] = NNdir + '/data'

if not os.getenv('smartNN_DATABASE_PATH'):
    os.environ['smartNN_DATABASE_PATH'] = NNdir + '/database'

if not os.getenv('smartNN_SAVE_PATH'):
    os.environ['smartNN_SAVE_PATH'] = NNdir + '/save'

print('smartNN_DATA_PATH = ' + os.environ['smartNN_DATA_PATH'])
print('smartNN_SAVE_PATH = ' + os.environ['smartNN_SAVE_PATH'])
print('smartNN_DATABASE_PATH = ' + os.environ['smartNN_DATABASE_PATH'])



def autoencoder():
    
    learning_rule = LearningRule(max_col_norm = None,
                            learning_rate = 0.01,
                            momentum = 0.1,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='entropy'),
                            stopping_criteria = {'max_epoch' : 10,
                                                'cost' : Cost(type='entropy'),
                                                'epoch_look_back' : 3,
                                                'percent_decrease' : 0.001}
                            )
    
    data = Mnist()
    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)
    
    mlp = MLP(input_dim = data.feature_size(), rand_seed=None)
    h1_layer = RELU(dim=100, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    mlp.add_layer(Sigmoid(dim=data.target_size(), name='output_layer', W=h1_layer.W.T, b=None))

    log = Log(experiment_id = 'AE',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()
    
def stacked_autoencoder():

    name = 'stacked_AE4'

    #=====[ Train First layer of stack autoencoder ]=====#
    print('Start training First Layer of AutoEncoder')

    
    log = Log(experiment_id = name + '_layer1',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    learning_rule = LearningRule(max_col_norm = None,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                weight_decay = 0,
                                cost = Cost(type='entropy'),
                                stopping_criteria = {'max_epoch' : 1000, 
                                                    'epoch_look_back' : 10,
                                                    'cost' : Cost(type='entropy'), 
                                                    'percent_decrease' : 0.001}
                                )

    data = Mnist()
                    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)
    
    mlp = MLP(input_dim = data.feature_size(), rand_seed=None)

    h1_layer = Sigmoid(dim=500, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    h1_mirror = Sigmoid(dim = data.target_size(), name='h1_mirror', W=h1_layer.W.T, b=None)
    mlp.add_layer(h1_mirror)

    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()
    
    #=====[ Train Second Layer of autoencoder ]=====#
    
    log2 = Log(experiment_id = name + '_layer2',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    learning_rule = LearningRule(max_col_norm = None,
                            learning_rate = 0.01,
                            momentum = 0.1,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='entropy'),
                            stopping_criteria = {'max_epoch' : 1000, 
                                                'epoch_look_back' : 10,
                                                'cost' : Cost(type='entropy'), 
                                                'percent_decrease' : 0.001}
                            )

    
    print('Start training Second Layer of AutoEncoder')
    
    mlp.pop_layer(-1)
    reduced_train_X = np.abs(mlp.fprop(train.X))
    reduced_valid_X = np.abs(mlp.fprop(valid.X))
    reduced_test_X = np.abs(mlp.fprop(test.X))

    data.set_train(reduced_train_X, reduced_train_X)
    data.set_valid(reduced_valid_X, reduced_valid_X)
    data.set_test(reduced_test_X, reduced_test_X)
    
    # create a new mlp taking inputs from the encoded outputs of first autoencoder
    mlp2 = MLP(input_dim = data.feature_size(), rand_seed=None)

    
    h2_layer = Sigmoid(dim=100, name='h2_layer', W=None, b=None)
    mlp2.add_layer(h2_layer)
    
    h2_mirror = Sigmoid(dim=h1_layer.dim, name='h2_mirror', W=h2_layer.W.T, b=None)
    mlp2.add_layer(h2_mirror)
    
              
    train_object = TrainObject(model = mlp2,
                            dataset = data,
                            learning_rule = learning_rule,
                            log = log2)
    
    train_object.run()
    
    #=====[ Fine Tuning ]=====#
    
    log3 = Log(experiment_id = name + '_full',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    print('Fine Tuning')
    
    data = Mnist()
    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)
    
    mlp3 = MLP(input_dim = data.feature_size(), rand_seed=None)
    mlp3.add_layer(h1_layer)
    mlp3.add_layer(h2_layer)
    mlp3.add_layer(h2_mirror)
    mlp3.add_layer(h1_mirror)
    
    
    train_object = TrainObject(model = mlp3,
                            dataset = data,
                            learning_rule = learning_rule,
                            log = log3)
    
    train_object.run()
    print('..Training Done')