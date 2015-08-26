
import numpy as np

import theano.tensor as T

from mozi.datasets.voc import VOC
from mozi.model import Sequential
from mozi.layers.alexnet import Alexnet
from mozi.log import Log
from mozi.train_object import TrainObject
from mozi.cost import error, entropy
from mozi.learning_method import SGD
import os

import theano

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

    data = VOC(batch_size=32, train_valid_test_ratio=[5,1,1])
    model = Sequential(input_var=T.tensor4(), output_var=T.matrix())
    model.add(Alexnet(input_shape=(3,222,222), output_dim=11))
    # build learning method
    learning_method = SGD(learning_rate=0.01, momentum=0.9,
                          lr_decay_factor=0.9, decay_batch=5000)
    # put everything into the train object
    train_object = TrainObject(model = model,
                               log = None,
                               dataset = data,
                               train_cost = error,
                               valid_cost = error,
                               learning_method = learning_method,
                               stop_criteria = {'max_epoch' : 10,
                                                'epoch_look_back' : 5,
                                                'percent_decrease' : 0.01}
                               )
    # finally run the code
    train_object.setup()
    train_object.run()

if __name__ == '__main__':
    setenv()
    train()
