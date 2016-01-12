import numpy as np
import os
import theano.tensor as T

from mozi.datasets.cifar10 import Cifar10
from mozi.model import Sequential
from mozi.layers.linear import *
from mozi.layers.noise import Dropout
from mozi.layers.activation import *
from mozi.layers.convolution import *
from mozi.layers.misc import Flatten
from mozi.log import Log
from mozi.train_object import TrainObject
from mozi.cost import error, entropy
from mozi.learning_method import SGD
from mozi.datasets.dataset import DataBlocks

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
    # create a fake dataset
    X1 = np.random.rand(1000000, 1000)
    y1 = np.random.rand(1000000, 10)
    with open('X1.npy', 'wb') as xin, open('y1.npy', 'wb') as yin:
        np.save(xin, X1)
        np.save(yin, y1)

    X2 = np.random.rand(1000000, 1000)
    y2 = np.random.rand(1000000, 10)
    with open('X2.npy', 'wb') as xin, open('y2.npy', 'wb') as yin:
        np.save(xin, X2)
        np.save(yin, y2)

    X3 = np.random.rand(1000000, 1000)
    y3 = np.random.rand(1000000, 10)
    with open('X3.npy', 'wb') as xin, open('y3.npy', 'wb') as yin:
        np.save(xin, X3)
        np.save(yin, y3)

    # now we can create the data by putting the paths
    # ('X1.npy', 'y1.npy') and ('X2.npy', 'y2.npy') into DataBlocks
    data = DataBlocks(data_paths=[('X1.npy', 'y1.npy'), ('X2.npy', 'y2.npy'), ('X3.npy', 'y3.npy')],
                      batch_size=100, train_valid_test_ratio=[3,2,0])


    model = Sequential(input_var=T.matrix(), output_var=T.matrix())
    model.add(Linear(prev_dim=1000, this_dim=200))
    model.add(RELU())
    model.add(Linear(prev_dim=200, this_dim=100))
    model.add(RELU())
    model.add(Dropout(0.5))
    model.add(Linear(prev_dim=100, this_dim=10))
    model.add(Softmax())

    # build learning method
    learning_method = SGD(learning_rate=0.01, momentum=0.9,
                          lr_decay_factor=0.9, decay_batch=5000)

    # put everything into the train object
    train_object = TrainObject(model = model,
                               log = None,
                               dataset = data,
                               train_cost = entropy,
                               valid_cost = error,
                               learning_method = learning_method,
                               stop_criteria = {'max_epoch' : 10,
                                                'epoch_look_back' : 5,
                                                'percent_decrease' : 0.01}
                               )
    # finally run the code
    train_object.setup()
    train_object.run()

    for X_path, y_path in [('X1.npy', 'y1.npy'), ('X2.npy', 'y2.npy')]:
        with open(X_path) as Xin, open(y_path) as yin:
            # test the model on test set
            ypred = model.fprop(np.load(Xin))
            ypred = np.argmax(ypred, axis=1)
            y = np.argmax(np.load(yin), axis=1)
            accuracy = np.equal(ypred, y).astype('f4').sum() / len(y)
            print 'combined accuracy for blk %s:'%X_path, accuracy


if __name__ == '__main__':
    setenv()
    train()
