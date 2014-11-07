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


class SGD(LearningMethod):

    def __init__(self, learning_rate, momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update(self, delta, gparam):
        return [(delta, self.momentum * delta - self.learning_rate * gparam)]


class AdaGrad(LearningMethod):

    def __init__(self, learning_rate=0.9, k=1):
        """
        dx = -learning_rate / sqrt(k + sum(gparam^2)) * gparam
        ref : Chris Dyer : Notes on AdaGrad
        """
        self.learning_rate = learning_rate
        self.k = k

    def update(self, delta, gparam):
        # eps = self.k * T.ones_like(delta)
        eps = theano.shared(self.k * np.ones(delta.shape.eval(), dtype=floatX))
        rlist = []
        rlist.append((eps, eps + gparam ** 2))
        rlist.append((delta, -self.learning_rate * gparam / T.sqrt(eps)))
        return rlist
        

class AdaDelta(LearningMethod):

    def __init__(self, eps=1e-6, rho=0.95):
        """
        dx_t = -rms(dx_{t-1}) / rms(gparam_t) * gparam_t
        rms(dx) = sqrt(E_t(dx^2) + eps)
        E_t(dx^s) = rho E_{t-1}(dx^2) + (1-rho) dx^2
        ref : Matthew D. Zeiler: ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
        """
        self.eps = eps
        self.rho = rho

    def update(self, delta, gparam):
        rlist = []
        # gparam_mean = T.zeros_like(gparam)
        gparam_mean = theano.shared(np.zeros(delta.shape.eval(), dtype=floatX))
        rlist.append((gparam_mean, self.rho * gparam_mean + (1-self.rho) * gparam**2))
        # delta_mean = T.zeros_like(delta)
        delta_mean = theano.shared(np.zeros(delta.shape.eval(), dtype=floatX))
        rlist.append((delta_mean, self.rho * delta_mean + (1-self.rho) * delta**2))
        rlist.append((delta, -T.sqrt(delta_mean+self.eps) / T.sqrt(gparam_mean+self.eps) * gparam))
        return rlist
