import logging
logger = logging.getLogger(__name__)
import os
import cPickle
import numpy as np
import theano
floatX = theano.config.floatX

from mozi.utils.utils import get_file, make_one_hot
from mozi.datasets.dataset import SingleBlock

class Cifar100(SingleBlock):

    def __init__(self, flatten=False, fine_label=True, **kwargs):
        '''
        PARAM:
            fine_label: True (100 classes) False (20 classes)
        '''

        im_dir = os.environ['MOZI_DATA_PATH'] + '/cifar100/'
        path = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        im_dir = get_file(fpath="{}/cifar-100-python.tar.gz".format(im_dir), origin=path, untar=True)

        self.img_shape = (3,32,32)
        self.img_size = np.prod(self.img_shape)

        fnames = ['train', 'test']

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
                if fine_label:
                    y.extend(data_batch['fine_labels'])
                    self.n_classes = 100
                else:
                    y.extend(data_batch['coarse_labels'])
                    self.n_classes = 20

        X_npy = np.array(X, dtype=floatX)
        X_npy /= 255.0
        y_npy = make_one_hot(y, onehot_size=self.n_classes)

        super(Cifar100, self).__init__(X=X_npy, y=y_npy, **kwargs)
