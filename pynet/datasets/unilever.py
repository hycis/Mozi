import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano

from pynet.datasets.dataset import SingleBlock, DataBlocks
from pynet.utils.utils import make_one_hot

class Unilever(SingleBlock):

    def __init__(self, one_hot=False, **kwargs):
        dir = '/Volumes/Storage/Unilever_Challenge/dataset'
        with open(dir + '/train.npy') as Xin:
            data = np.load(Xin)

        X, y = self.make_Xy(data)
        if one_hot:
            y = make_one_hot(y, 8)
        else:
            y = y.reshape((y.shape[0], 1))

        super(Unilever, self).__init__(X=X, y=y, **kwargs)


    def make_Xy(self, data):
        X = data[:, 158:-2]
        y = data[:, -1]
        return X, y
