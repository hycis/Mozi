import os

import theano
import theano.tensor as T
import numpy as np

from mozi.datasets.mnist import Mnist
from mozi.datasets.preprocessor import *
from mozi.model import Sequential
from mozi.layers.linear import *
from mozi.layers.activation import *
from mozi.layers.noise import Dropout
from mozi.log import Log
from mozi.train_object import TrainObject
from mozi.cost import mse, error
from mozi.learning_method import *
from mozi.weight_init import *


def setenv():
    NNdir = os.path.dirname(os.path.realpath(__file__))
    NNdir = os.path.dirname(NNdir)

    # directory to save all the dataset
    if not os.getenv('MOZI_DATA_PATH'):
        os.environ['MOZI_DATA_PATH'] = NNdir + '/data'

    # directory for saving the database that is used for logging the results
    if not os.getenv('MOZI_DATABASE_PATH'):
        os.environ['MOZI_DATABASE_PATH'] = NNdir + '/database'

    # directory to save all the trained models and outputs
    if not os.getenv('MOZI_SAVE_PATH'):
        os.environ['MOZI_SAVE_PATH'] = NNdir + '/save'

    print('MOZI_DATA_PATH = ' + os.environ['MOZI_DATA_PATH'])
    print('MOZI_SAVE_PATH = ' + os.environ['MOZI_SAVE_PATH'])
    print('MOZI_DATABASE_PATH = ' + os.environ['MOZI_DATABASE_PATH'])


def train():

    # build dataset
    data = Mnist(batch_size=64, train_valid_test_ratio=[5,1,1])

    # build model
    model = Sequential()
    model.add(Linear(prev_dim=28*28, this_dim=200))
    model.add(RELU())
    model.add(Linear(prev_dim=200, this_dim=100))
    model.add(RELU())
    model.add(Dropout(0.5))
    model.add(Linear(prev_dim=100, this_dim=10))
    model.add(Softmax())

    # build learning method
    learning_method = AdaGrad(learning_rate=0.1, momentum=0.9,
                              lr_decay_factor=0.9, decay_batch=10000)

    # Build Logger
    log = Log(experiment_name = 'MLP',
              description = 'This is a tutorial',
              save_outputs = True, # log all the outputs from the screen
              save_model = True, # save the best model
              save_epoch_error = True, # log error at every epoch
              save_to_database = {'name': 'Example.db',
                                  'records': {'Batch_Size': data.batch_size,
                                              'Learning_Rate': learning_method.learning_rate,
                                              'Momentum': learning_method.momentum}}
             ) # end log

    # put everything into the train object
    train_object = TrainObject(model = model,
                               log = log,
                               dataset = data,
                               train_cost = mse,
                               valid_cost = error,
                               learning_method = learning_method,
                               stop_criteria = {'max_epoch' : 10,
                                                'epoch_look_back' : 5,
                                                'percent_decrease' : 0.01}
                               )
    # finally run the code
    train_object.setup()
    train_object.run()

    ypred = model.fprop(data.get_test().X)
    ypred = np.argmax(ypred, axis=1)
    y = np.argmax(data.get_test().y, axis=1)
    accuracy = np.equal(ypred, y).astype('f4').sum() / len(y)
    print 'test accuracy:', accuracy


if __name__ == '__main__':
    setenv()
    train()
