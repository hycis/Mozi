import logging
logger = logging.getLogger(__name__)
import os
import cPickle
import numpy as np
import theano
floatX = theano.config.floatX

from pynet.utils.mnist_ubyte import read_mnist_images
from pynet.utils.mnist_ubyte import read_mnist_labels
from pynet.datasets.dataset import SingleBlock

class Cifar10(SingleBlock):

    def __init__(self, **kwargs):

        im_dir = os.environ['PYNET_DATA_PATH'] + '/cifar10/'
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog','horse','ship','truck']
        self.img_shape = (3,32,32)
        self.img_size = np.prod(self.img_shape)
        self.n_classes = 10
        fnames = ['data_batch_%i' % i for i in range(1,6)]

        X = []
        y = []
        for fname in fnames:
            data_path = im_dir + fname
            with open(data_path) as fin:
                data_batch = cPickle.load(fin)
                X.extend(data_batch['data'].tolist())
                y.extend(data_batch['labels'])
        X_npy = np.array(X, dtype=floatX)
        y_npy = np.zeros((len(y), 10), dtype=floatX)
        for i in xrange(len(y)):
            y_npy[i, y_npy[i]] = 1

        super(Cifar10, self).__init__(X=X_npy, y=y_npy, **kwargs)
