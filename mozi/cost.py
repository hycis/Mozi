__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"

import theano.tensor as T
import theano
from mozi.utils.utils import theano_unique

floatX = theano.config.floatX

def get_accuracy(y, y_pred):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label
    """

    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', y_pred.type))

    return T.eq(y_pred.argmax(axis=1), y.argmax(axis=1)).sum() / y.shape[0]

def positives(y, y_pred):
    """
    return the number of correctly predicted examples in a batch
    """
    return T.eq(y_pred.argmax(axis=1), y.argmax(axis=1)).sum()

def _batch_cost_nll(y, y_pred):
    """
    return the total cost of all the examples in a batch
    """
    return T.sum(T.log(y_pred)[T.arange(y.shape[0]), y.argmin(axis=1)])

def confusion_matrix(y, y_pred):
    #TODO
    pass

def mse(y, y_pred):
    L = T.sum(T.sqr(y - y_pred), axis=1)
    return T.mean(L)

def entropy(y, y_pred):
    L = - T.sum(y * T.log(y_pred) + (1-y) * T.log(1-y_pred), axis=1)
    return T.mean(L)

def error(y, y_pred):
    L = T.neq(y_pred.argmax(axis=1), y.argmax(axis=1))
    return T.mean(L)

def f1(y, y_pred):
    #TODO
    pass

def binary_misprecision(y, y_pred):
    '''
    This cost function is only for binary classifications
    '''
    # assert(theano_unique(y).size == 2)

    y_pred = y_pred.argmax(axis=1)
    y = y.argmax(axis=1)

    TP = (y_pred and y).astype(floatX)
    y0 = T.eq(y, 0)
    FP = (y0 and y_pred).astype(floatX)

    TP = T.sum(TP)
    FP = T.sum(FP)

    rval = FP / (TP + FP)
    return rval

def FP_minus_TP(y, y_pred):
    '''
    This cost function is only for binary classifications
    '''
    # assert(theano_unique(y).size == 2)

    y_pred = y_pred.argmax(axis=1)
    y = y.argmax(axis=1)

    TP = (y_pred and y).astype(floatX)
    y0 = T.eq(y, 0)
    FP = (y0 and y_pred).astype(floatX)

    TP = T.mean(TP)
    FP = T.mean(FP)

    return FP - TP


def abs(y, y_pred):
    L = T.sum(T.abs_(y - y_pred, axis=1))
    return T.mean(L)
