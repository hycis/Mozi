
import os

import theano
import theano.tensor as T
import numpy as np

from mozi.datasets.mnist import Mnist
from mozi.model import Sequential
from mozi.layers.vae import VariationalAutoencoder
from mozi.log import Log
from mozi.train_object import TrainObject
from mozi.cost import SGVB_bin
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
    """
    This examples implements the variational autoencoder from the paper
    Auto-Encoding Variational Bayes by Diederik P Kingma, Max Welling, arXiv:1312.6114
    """

    # build dataset
    data = Mnist(batch_size=100, binary=False, train_valid_test_ratio=[5,1,1])
    # for autoencoder, the output will be equal to input
    data.set_train(X=data.get_train().X, y=data.get_train().X)
    data.set_valid(X=data.get_valid().X, y=data.get_valid().X)

    # build model
    model = Sequential(input_var=T.matrix(), output_var=T.matrix())
    model.add(VariationalAutoencoder(input_dim=28*28, bottlenet_dim=200, z_dim=20))

    # build learning method
    learning_method = SGD(learning_rate=0.0001, momentum=0.9,
                              lr_decay_factor=0.9, decay_batch=10000)

    # put everything into the train object
    train_object = TrainObject(model = model,
                               log = None,
                               dataset = data,
                               train_cost = SGVB_bin,
                               valid_cost = SGVB_bin,
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
