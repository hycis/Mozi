
import theano.tensor as T
import theano
from mozi.utils.utils import theano_unique
from mozi.utils.theano_utils import asfloatX

floatX = theano.config.floatX

if floatX == 'float64':
    epsilon = 1.0e-8
else:
    epsilon = 1.0e-6

def accuracy(y, y_pred):
    L = T.eq(y_pred.argmax(axis=1), y.argmax(axis=1))
    return T.sum(L) / y.shape[0].astype(floatX)

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

def recall(y, y_pred):
    L = T.eq(y_pred.argmax(axis=1), y.argmax(axis=1))
    return T.sum(L) / y.shape[0].astype(floatX)

def precision(y, y_pred):
    L = T.eq(y_pred.argmax(axis=1), y.argmax(axis=1))
    return T.sum(L) / y_pred.shape[0].astype(floatX)

def f1(y, y_pred):
    r = recall(y, y_pred)
    p = precision(y, y_pred)
    return 2 * p * r / (p + r)

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
