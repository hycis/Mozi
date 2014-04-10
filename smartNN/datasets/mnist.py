import os
import numpy as np
import theano

from smartNN.utils.mnist_ubyte import read_mnist_images
from smartNN.utils.mnist_ubyte import read_mnist_labels
from smartNN.datasets.dataset import IterMatrix, Dataset

class Mnist(Dataset):
    
    def __init__(self, binarize=False, train_ratio=5, valid_ratio=1, **kwargs):
                
        im_dir = os.environ['smartNN_DATA_PATH'] + '/mnist/'
        
        train_X = read_mnist_images(im_dir + 'train-images-idx3-ubyte', dtype='float32')
        train_y = read_mnist_labels(im_dir + 'train-labels-idx1-ubyte')
        
        test_X = read_mnist_images(im_dir + 't10k-images-idx3-ubyte', dtype='float32')
        test_y = read_mnist_labels(im_dir + 't10k-labels-idx1-ubyte')
        
        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])
        test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2])

        train_y_tmp = np.zeros((train_X.shape[0], 10), dtype=theano.config.floatX)
        test_y_tmp = np.zeros((test_X.shape[0], 10), dtype=theano.config.floatX)
        
        for i in xrange(train_X.shape[0]):
            train_y_tmp[i, train_y[i]] = 1
        
        for i in xrange(test_X.shape[0]):
            test_y_tmp[i, test_y[i]] = 1
        
        
        train_y = train_y_tmp
        test_y = test_y_tmp
        
        num_examples = train_X.shape[0]
        num_valid = np.floor(valid_ratio * 1.0 / (valid_ratio + train_ratio) 
                                * num_examples).astype('int')
                
        valid_X = train_X[:num_valid]
        valid_y = train_y[:num_valid]
        
        train_X = train_X[num_valid:]
        train_y = train_y[num_valid:]
                

        super(Mnist, self).__init__(train=[train_X, train_y], 
                                    valid=[valid_X, valid_y],
                                    test=[test_X, test_y], **kwargs)

        
     