import os

import theano
import theano.tensor as T
import numpy as np

from pynet.model import MLP
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
    os.environ['PYNET_DATA_PATH'] = NNdir + '/data'

if not os.getenv('PYNET_DATABASE_PATH'):
    os.environ['PYNET_DATABASE_PATH'] = NNdir + '/database'

if not os.getenv('PYNET_SAVE_PATH'):
    os.environ['PYNET_SAVE_PATH'] = NNdir + '/save'

print('PYNET_DATA_PATH = ' + os.environ['PYNET_DATA_PATH'])
print('PYNET_SAVE_PATH = ' + os.environ['PYNET_SAVE_PATH'])
print('PYNET_DATABASE_PATH = ' + os.environ['PYNET_DATABASE_PATH'])


def mlp():

    data = Mnist(train_valid_test_ratio=[5,1,1])


    mlp = MLP(input_dim = data.feature_size())

    mlp.add_layer(Sigmoid(dim=200, name='h1_layer', W=None, b=None, dropout_below=0.5))
    mlp.add_layer(Sigmoid(dim=data.target_size(), name='output_layer', W=None, b=None, dropout_below=0.5))

    learning_rule = LearningRule(max_col_norm = 10,
                                learning_rate = 0.01,
                                momentum = 0.,
                                momentum_type = 'normal',
                                L1_lambda = None,
                                L2_lambda = None,
                                training_cost = Cost(type='mse'),
                                stopping_criteria = {'max_epoch' : 100,
                                                    'epoch_look_back' : 3,
                                                    'cost' : Cost(type='error'),
                                                    'percent_decrease' : 0.001}
                                )

    log = Log(experiment_name = 'mnistest2',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = False,
            save_model = True,
            save_to_database = {'name': 'Database.db',
                                'records' : {'Dataset' : data.__class__.__name__,
                                             'Weight_Init_Seed' : mlp.rand_seed,
                                             'Dropout_Below' : str([layer.dropout_below for layer in mlp.layers]),
                                             'Batch_Size' : data.batch_size,
                                             'Layer_Size' : len(mlp.layers),
                                             'Layer_Dim' : str([layer.dim for layer in mlp.layers]),
                                             'Preprocessor' : data.preprocessor.__class__.__name__,
                                             'Learning_Rate' : learning_rule.learning_rate,
                                             'Momentum' : learning_rule.momentum}}
            ) # end log

    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
    train_object.run()

if __name__ == '__main__':
    mlp()
