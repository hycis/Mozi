import numpy as np

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


from sklearn.metrics import accuracy_score

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

    data = Cifar10(batch_size=32, train_valid_test_ratio=[5,1,1])

    model = Sequential()
    model.add(Convolution2D(input_channels=3, filters=32, kernel_size=(3,3), stride=(1,1), border_mode='full'))
    model.add(RELU())
    model.add(Convolution2D(input_channels=32, filters=32, kernel_size=(3,3), stride=(1,1)))
    model.add(RELU())
    model.add(Pooling2D(poolsize=(2, 2), mode='max'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(input_channels=32, filters=64, kernel_size=(3,3), stride=(1,1), border_mode='full'))
    model.add(RELU())
    model.add(Convolution2D(input_channels=64, filters=64, kernel_size=(3,3), stride=(1,1),))
    model.add(RELU())
    model.add(Pooling2D(poolsize=(2, 2), mode='max'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Linear(64*8*8, 512))
    model.add(RELU())
    model.add(Dropout(0.5))

    model.add(Linear(512, 10))
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

    # test the model on test set
    ypred = model.fprop(data.get_test().X)
    ypred = np.argmax(ypred, axis=1)
    y = np.argmax(data.get_test().y, axis=1)
    print 'test accuracy:', accuracy_score(y, ypred)


if __name__ == '__main__':
    setenv()
    train()
