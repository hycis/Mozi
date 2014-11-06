import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano

from pynet.utils.mnist_ubyte import read_mnist_images
from pynet.utils.mnist_ubyte import read_mnist_labels
from pynet.datasets.dataset import SingleBlock, DataBlocks

class Mnist(SingleBlock):

    def __init__(self, **kwargs):

        im_dir = os.environ['PYNET_DATA_PATH'] + '/mnist/'

        train_X = read_mnist_images(im_dir + 'train-images-idx3-ubyte', dtype='float32')
        train_y = read_mnist_labels(im_dir + 'train-labels-idx1-ubyte')

        test_X = read_mnist_images(im_dir + 't10k-images-idx3-ubyte', dtype='float32')
        test_y = read_mnist_labels(im_dir + 't10k-labels-idx1-ubyte')

        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])
        test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2])

        X = np.concatenate((train_X, test_X), axis=0)

        train_y_tmp = np.zeros((train_X.shape[0], 10), dtype=theano.config.floatX)
        test_y_tmp = np.zeros((test_X.shape[0], 10), dtype=theano.config.floatX)

        for i in xrange(train_X.shape[0]):
            train_y_tmp[i, train_y[i]] = 1

        for i in xrange(test_X.shape[0]):
            test_y_tmp[i, test_y[i]] = 1

        train_y = train_y_tmp
        test_y = test_y_tmp

        y = np.concatenate((train_y, test_y), axis=0)

        super(Mnist, self).__init__(X=X, y=y, **kwargs)

class Mnist_Blocks(DataBlocks):

    def __init__(self, feature_size=784, target_size=10, **kwargs):

        # self.parts = [ 'blk1.npy', 'blk2.npy']
        parts = ['fullblk.npy']
        data_dir = os.environ['PYNET_DATA_PATH'] + '/mnist_npy'
        data_paths = ["%s/%s"%(data_dir, part) for part in parts]
        super(Mnist_Blocks, self).__init__(data_paths=data_paths,
                                           feature_size=feature_size,
                                           target_size=target_size, **kwargs)


    # def __iter__(self):
    #     self.files = iter(self.parts)
    #     return self
    #
    # def next(self):
    #     self.dataset.train = None
    #     self.dataset.valid = None
    #     self.dataset.test = None
    #     with open(self.data_dir + '/' + next(self.files), 'rb') as f:
    #         data = np.load(f)
    #     if self.dataset.preprocessor is not None:
    #         logger.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
    #         data = self.dataset.preprocessor.apply(data)
    #     self.dataset.set_Xy(X=data, y=data)
    #     data = None
    #     return self.dataset
    #
    # def nblocks(self):
    #     return len(self.parts)

class Mnist_Blocks_500(Mnist_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):

        # self.parts = [ 'blk1.npy', 'blk2.npy']
        # self.parts = ['fullblk.npy']
        super(Mnist_Blocks_500, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/AE_Testing_Mnist_784_500_20140908_2215_15217959'
