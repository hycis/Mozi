import os

import theano
import theano.tensor as T
import numpy as np

from smartNN.model import MLP
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


def mlp():
    
    data = Mnist(train_valid_test_ratio=[5,1,1])
    

    mlp = MLP(input_dim = data.feature_size())
    
    mlp.add_layer(Sigmoid(dim=100, name='h1_layer', W=None, b=None, dropout_below=None))
    mlp.add_layer(Sigmoid(dim=data.target_size(), name='output_layer', W=None, b=None, dropout_below=None))
    
    learning_rule = LearningRule(max_col_norm = 1,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                L1_lambda = None,
                                L2_lambda = None,
                                cost = Cost(type='mse'),
                                stopping_criteria = {'max_epoch' : 100, 
                                                    'epoch_look_back' : 3,
                                                    'cost' : Cost(type='error'), 
                                                    'percent_decrease' : 0.001}
                                )
    
    log = Log(experiment_id = 'mnistest2',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = False,
            save_model = False,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
    train_object.run()
    
if __name__ == '__main__':
    mlp()