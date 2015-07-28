import logging
logger = logging.getLogger(__name__)
import os
import cPickle
import numpy as np
import theano
floatX = theano.config.floatX

from mozi.utils.utils import get_file, make_one_hot
from mozi.datasets.dataset import SingleBlock

class Cifar10(SingleBlock):

    def __init__(self, flatten=False, **kwargs):

        im_dir = os.environ['MOZI_DATA_PATH'] + '/cifar10/'
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog','horse','ship','truck']
        path = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        im_dir = get_file(fpath="{}/cifar-10-python.tar.gz".format(im_dir), origin=path, untar=True)

        self.img_shape = (3,32,32)
        self.img_size = np.prod(self.img_shape)
        self.n_classes = 10
        fnames = ['data_batch_%i' % i for i in range(1,6)] + ['test_batch']

        X = []
        y = []
        for fname in fnames:
            data_path = "{}/{}".format(im_dir, fname)
            with open(data_path) as fin:
                data_batch = cPickle.load(fin)
                if flatten:
                    X.extend(data_batch['data'].reshape((len(data_batch['data']), self.img_size)))
                else:
                    X.extend(data_batch['data'].reshape((len(data_batch['data']),)+self.img_shape))
                y.extend(data_batch['labels'])


        X_npy = np.array(X, dtype=floatX)
        X_npy /= 255.0
        y_npy = make_one_hot(y, onehot_size=self.n_classes)

        super(Cifar10, self).__init__(X=X_npy, y=y_npy, **kwargs)
