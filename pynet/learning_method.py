__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"


import theano
import theano.tensor as T
floatX = theano.config.floatX

import numpy as np

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

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = theano.shared(np.asarray(learning_rate, dtype=floatX))
        self.mom = theano.shared(np.asarray(momentum, dtype=floatX))

    def update(self, delta, gparam):
        return [(delta, self.mom * delta - self.lr * gparam)]


class AdaGrad(LearningMethod):

    def __init__(self, learning_rate=0.9, momentum=0., k=1.0):
        """
        dx = -learning_rate / sqrt(k + sum(gparam^2)) * gparam
        ref : Chris Dyer : Notes on AdaGrad
        """
        self.lr = theano.shared(np.asarray(learning_rate, dtype=floatX))
        self.mom = theano.shared(np.asarray(momentum, dtype=floatX))
        self.k = theano.shared(np.asarray(k, dtype=floatX))

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
        self.eps = theano.shared(np.asarray(eps, dtype=floatX))
        self.rho = theano.shared(np.asarray(rho, dtype=floatX))

    def update(self, delta, gparam):
        rlist = []
        gparam_mean = theano.shared(np.zeros_like(delta.get_value(borrow=True, return_internal_type=True)))
        rlist.append((gparam_mean, self.rho * gparam_mean + (1-self.rho) * gparam**2))
        delta_mean = theano.shared(np.zeros_like(delta.get_value(borrow=True, return_internal_type=True)))
        rlist.append((delta_mean, self.rho * delta_mean + (1-self.rho) * delta**2))
        rlist.append((delta, -T.sqrt(delta_mean+self.eps) / T.sqrt(gparam_mean+self.eps) * gparam))
        return rlist
