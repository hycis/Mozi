
import theano.tensor as T
import theano
from mozi.utils.utils import theano_unique

floatX = theano.config.floatX

if floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7

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

def mse(y, y_pred):
    L = T.sum(T.sqr(y - y_pred), axis=1)
    return T.mean(L)

def entropy(y, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    L = - T.sum(y * T.log(y_pred) + (1-y) * T.log(1-y_pred), axis=1)
    return T.mean(L)

def error(y, y_pred):
    L = T.neq(y_pred.argmax(axis=1), y.argmax(axis=1))
    return T.mean(L)

def f1(y, y_pred):
    #TODO
    pass

def abs(y, y_pred):
    L = T.sum(T.abs_(y - y_pred, axis=1))
    return T.mean(L)

def SGVB_bin(y, y_pred):
    '''
    This cost function for variational autoencoder for binary inputs
    '''
    ypred, miu_e, logsig_e = y_pred
    ypred = T.clip(ypred, epsilon, 1.0 - epsilon)
    logpxz = -T.nnet.binary_crossentropy(ypred, y).sum(axis=1)
    L = logpxz + 0.5 * (1 + 2*logsig_e - miu_e**2 - T.exp(2*logsig_e)).sum(axis=1)
    return L.mean()
