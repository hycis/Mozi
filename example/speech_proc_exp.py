
from smartNN.mlp import MLP
from smartNN.layer import RELU, Sigmoid, Softmax, Linear
from smartNN.datasets.mnist import Mnist
from smartNN.learning_rule import LearningRule
from smartNN.log import Log
from smartNN.train_object import TrainObject
from smartNN.cost import Cost

from smartNN.datasets.spec import P276_Spec

import os

import theano
import theano.tensor as T
import numpy as np

NNdir = os.path.dirname(os.path.realpath(__file__))

# if not os.getenv('smartNN_DATA_PATH'):
os.environ['smartNN_DATA_PATH'] = '/Applications/VCTK/data'

if not os.getenv('smartNN_SAVE_PATH'):
    os.environ['smartNN_SAVE_PATH'] = NNdir + '/save'

print('smartNN_DATA_PATH = ' + os.environ['smartNN_DATA_PATH'])
print('smartNN_SAVE_PATH = ' + os.environ['smartNN_SAVE_PATH'])


def mlp():
     
    data = P276_Spec(preprocess = None, 
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    test_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
    
    mlp = MLP(input_dim = data.feature_size())
    mlp.add_layer(RELU(dim=100, name='h1_layer', W=None, b=None))
    mlp.add_layer(RELU(dim= data.target_size(), name='output_layer', W=None, b=None))
    
    learning_rule = LearningRule(max_norm = 0.5,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                weight_decay = 0,
                                cost = Cost(type='mse'),
                                dropout_below = 1,
                                stopping_criteria = {'max_epoch' : 100, 
                                                    'epoch_look_back' : 3, 
                                                    'accu_increase' : 0.001}
                                )
    
    log = Log(experiment_id = 'testing',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
    train_object.run()


def spec_autoencoder():
    
    learning_rule = LearningRule(max_norm = 1,
                            learning_rate = 0.1,
                            momentum = 0.01,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='mse'),
                            dropout_below = 0,
                            stopping_criteria = {'max_epoch' : 10, 
                                                'epoch_look_back' : None, 
                                                'accu_increase' : None}
                            )
    
    data = P276_Spec(preprocess = None, 
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    test_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    data.valid = None
    data.test = None
    
#     valid = data.get_valid()
#     data.set_valid(valid.X, valid.X)
#     
#     test = data.get_test()
#     data.set_test(test.X, test.X)
    
    mlp = MLP(input_dim = data.feature_size(), rand_seed=None)
    h1_layer = RELU(dim=60, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    mlp.add_layer(Linear(dim=data.target_size(), name='output_layer', W=h1_layer.W.T, b=None))

    log = Log(experiment_id = 'testing',
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
    
    
def spec_stacked_AE():

    #=====[ Train First layer of stack autoencoder ]=====#
    print('Start training First Layer of AutoEncoder')

    
    log = Log(experiment_id = 'testing',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    learning_rule = LearningRule(max_norm = None,
                            learning_rate = 0.1,
                            momentum = 0.01,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='mse'),
                            dropout_below = 0,
                            stopping_criteria = {'max_epoch' : 10, 
                                                'epoch_look_back' : None, 
                                                'accu_increase' : None}
                            )

    data = P276_Spec(preprocess = None, 
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    test_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
                    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
#     valid = data.get_valid()
#     data.set_valid(valid.X, valid.X)
#     
#     test = data.get_test()
#     data.set_test(test.X, test.X)

    data.valid = None
    data.test = None
    
    mlp = MLP(input_dim = data.feature_size(), rand_seed=None)

    h1_layer = RELU(dim=200, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    h1_mirror = RELU(dim=data.target_size(), name='h1_mirror', W=h1_layer.W.T, b=None)
    mlp.add_layer(h1_mirror)

    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()
    
    #=====[ Train Second Layer of autoencoder ]=====#
    
    print('Start training Second Layer of AutoEncoder')
    
    x = T.matrix('x')
    mlp.pop_layer(-1)
    reduced_X = mlp.fprop(x)
    f = theano.function([x], reduced_X)
    reduced_X = f(train.X)
    
    data.set_train(reduced_X, reduced_X)
    data.valid = None
    data.test = None
    
    mlp2 = MLP(input_dim = data.feature_size(), rand_seed=None)

    
    h2_layer = RELU(dim=60, name='h2_layer', W=None, b=None)
    mlp2.add_layer(h2_layer)
    h2_mirror = Linear(dim=h1_layer.dim, name='h2_mirror', W=h2_layer.W.T, b=None)
    mlp2.add_layer(h2_mirror)
    
              
    train_object = TrainObject(model = mlp2,
                            dataset = data,
                            learning_rule = learning_rule,
                            log = log)
    
    train_object.run()
    
    #=====[ Fine Tuning ]=====#
    
    print('Fine Tuning')
    
    data = P276_spec(preprocess = None, 
                batch_size = 100,
                num_batches = None, 
                train_ratio = 5, 
                test_ratio = 1,
                iter_class = 'SequentialSubsetIterator')
    
    train = data.get_train()
    
    data.set_train(train.X, train.X)
    data.valid = None
    data.test = None
    
    mlp3 = MLP(input_dim = data.feature_size(), rand_seed=None)
    mlp3.add_layer(h1_layer)
    mlp3.add_layer(h2_layer)
    mlp3.add_layer(h2_mirror)
    mlp3.add_layer(h1_mirror)
    
    
    train_object = TrainObject(model = mlp3,
                            dataset = data,
                            learning_rule = learning_rule,
                            log = log)
    
    train_object.run()
        

    
if __name__ == '__main__':
    mlp()
#     spec_stacked_AE()   
