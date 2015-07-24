import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano

from mozi.utils.mnist_utils import read_mnist_images, read_mnist_labels, get_mnist_file
from mozi.datasets.dataset import SingleBlock, DataBlocks


class Mnist(SingleBlock):

    def __init__(self, **kwargs):

        im_dir = os.environ['MOZI_DATA_PATH'] + '/mnist/'

        url = 'http://yann.lecun.com/exdb/mnist'

        paths = []
        for fname in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                      't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']:
            path = get_mnist_file('{}/{}'.format(im_dir,fname), origin='{}/{}.gz'.format(url,fname))
            paths.append(path)

        train_X = read_mnist_images(paths[0], dtype='float32')
        train_y = read_mnist_labels(paths[1])

        test_X = read_mnist_images(paths[2], dtype='float32')
        test_y = read_mnist_labels(paths[3])

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

        self.parts = ['blk1.npy', 'blk2.npy']
        # parts = ['fullblk.npy']
        data_dir = os.environ['MOZI_DATA_PATH'] + '/mnist_npy'
        data_paths = ["%s/%s"%(data_dir, part) for part in parts]
        super(Mnist_Blocks, self).__init__(data_paths=data_paths,
                                           feature_size=feature_size,
                                           target_size=target_size, **kwargs)
