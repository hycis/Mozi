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
from mozi.layers.normalization import *
from mozi.log import Log
from mozi.train_object import TrainObject
from mozi.cost import error, entropy
from mozi.learning_method import *
from mozi.env import setenv
from mozi.utils.cnn_utils import valid, full

def train():
    batch_size = 128
    data = Cifar10(batch_size=batch_size, train_valid_test_ratio=[4,1,1])
    _, c, h, w = data.train.X.shape

    model = Sequential(input_var=T.tensor4(), output_var=T.matrix())
    model.add(Convolution2D(input_channels=c, filters=8, kernel_size=(3,3), stride=(1,1), border_mode='full'))
    h, w = full(h, w, kernel=3, stride=1)
    # model.add(BatchNormalization((8,h,w), short_memory=0.9))
    model.add(RELU())
    model.add(Convolution2D(input_channels=8, filters=16, kernel_size=(3,3), stride=(1,1), border_mode='valid'))
    h, w = valid(h, w, kernel=3, stride=1)
    # model.add(BatchNormalization((16,h,w), short_memory=0.9))
    model.add(RELU())
    model.add(Pooling2D(poolsize=(4, 4), stride=(4,4), mode='max'))
    h, w = valid(h, w, kernel=4, stride=4)
    model.add(Flatten())
    model.add(Linear(16*h*w, 512))
    model.add(BatchNormalization((512,), short_memory=0.9))
    model.add(RELU())

    model.add(Linear(512, 10))
    model.add(Softmax())

    learning_method = RMSprop(learning_rate=0.01)

    # Build Logger
    log = Log(experiment_name = 'cifar10_cnn_example',
              description = 'This is a tutorial',
              save_outputs = True, # log all the outputs from the screen
              save_model = True, # save the best model
              save_epoch_error = True, # log error at every epoch
              save_to_database = {'name': 'hyperparam.sqlite3',
                                  'records': {'Batch_Size': batch_size,
                                              'Learning_Rate': learning_method.learning_rate}}
             ) # end log

    # put everything into the train object
    train_object = TrainObject(model = model,
                               log = log,
                               dataset = data,
                               train_cost = entropy,
                               valid_cost = error,
                               learning_method = learning_method,
                               stop_criteria = {'max_epoch' : 30,
                                                'epoch_look_back' : 5,
                                                'percent_decrease' : 0.01}
                               )
    # finally run the code
    train_object.setup()
    train_object.run()

    # test the model on test set
    ypred = model.fprop(data.get_test().X)
    ypred = np.argmax(ypred, axis=1)
    y = np.argmax(data.get_test().y, axis=1)
    accuracy = np.equal(ypred, y).astype('f4').sum() / len(y)
    print 'test accuracy:', accuracy


if __name__ == '__main__':
    setenv()
    train()
