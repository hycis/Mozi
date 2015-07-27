
import theano
import theano.tensor as T
floatX = theano.config.floatX

import numpy as np
from mozi.utils.theano_utils import sharedX, asfloatX

class LearningMethod(object):

    def update(self, delta, gparam):
        """
        Return a list of tuples
        """
        raise NotImplementedError(str(type(self))+" does not implement delta.")

    @property
    def learning_rate(self):
        return float(self.lr.get_value())

    @property
    def momentum(self):
        return float(self.mom.get_value())

class SGD(LearningMethod):

    def __init__(self, learning_rate=0.01, momentum=0.9, lr_decay_factor=0.9, decay_batch=10000):
        self.lr = sharedX(learning_rate)
        self.mom = sharedX(momentum)
        self.batch = sharedX(0)
        self.decay_batch = sharedX(decay_batch)
        self.lr_decay_factor = asfloatX(lr_decay_factor)

    def update(self, delta, gparam):
        self.batch += 1
        if T.gt(self.batch, self.decay_batch):
            self.lr.set_value(self.lr.get_value() * self.lr_decay_factor)
            self.batch = sharedX(0)

        return [(delta, self.mom * delta - self.lr * gparam)]


class AdaGrad(LearningMethod):

    def __init__(self, learning_rate=0.9, momentum=0., k=1.0, lr_decay_factor=0.9, decay_batch=10000):
        """
        dx = -learning_rate / sqrt(k + sum(gparam^2)) * gparam
        ref : Chris Dyer : Notes on AdaGrad
        """
        self.lr = sharedX(learning_rate)
        self.mom = sharedX(momentum)
        self.k = sharedX(k)

    def update(self, delta, gparam):
        rlist = []
        eps = theano.shared(self.k.get_value() * np.ones_like(delta.get_value(borrow=True, return_internal_type=True)))
        rlist.append((eps, eps + gparam ** 2))
        rlist.append((delta, self.mom * delta - self.lr * gparam / T.sqrt(eps)))
        return rlist

class AdaDelta(LearningMethod):

    def __init__(self, eps=1e-6, rho=0.95):
        """
        dx_t = -rms(dx_{t-1}) / rms(gparam_t) * gparam_t
        rms(dx) = sqrt(E_t(dx^2) + eps)
        E_t(dx^s) = rho E_{t-1}(dx^2) + (1-rho) dx^2
        ref : Matthew D. Zeiler: ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
        """
        self.eps = sharedX(eps)
        self.rho = sharedX(rho)

    def update(self, delta, gparam):
        rlist = []
        gparam_mean = theano.shared(np.zeros_like(delta.get_value(borrow=True, return_internal_type=True)))
        rlist.append((gparam_mean, self.rho * gparam_mean + (1-self.rho) * gparam**2))
        delta_mean = theano.shared(np.zeros_like(delta.get_value(borrow=True, return_internal_type=True)))
        rlist.append((delta_mean, self.rho * delta_mean + (1-self.rho) * delta**2))
        rlist.append((delta, -T.sqrt(delta_mean+self.eps) / T.sqrt(gparam_mean+self.eps) * gparam))
        return rlist
