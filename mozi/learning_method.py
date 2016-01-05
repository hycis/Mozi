
import theano
import theano.tensor as T
from theano.ifelse import ifelse
floatX = theano.config.floatX

import numpy as np
from mozi.utils.theano_utils import sharedX, asfloatX

class LearningMethod(object):

    def update(self, deltas, params, gparams):
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

    def update(self, deltas, params, gparams):
        updates = []
        for delta, param, gparam in zip(deltas, params, gparams):
            updates.append((delta, self.mom * delta - self.lr * gparam))
            updates.append((param, param+delta))

        new_batch = ifelse(T.gt(self.batch, self.decay_batch), sharedX(0), self.batch+1)
        new_lr = ifelse(T.gt(self.batch, self.decay_batch), self.lr*self.lr_decay_factor, self.lr)
        updates.append((self.batch, new_batch))
        updates.append((self.lr, new_lr))
        return updates


class AdaGrad(LearningMethod):

    def __init__(self, learning_rate=0.9, momentum=0., k=1.0, lr_decay_factor=0.9, decay_batch=10000):
        """
        dx = -learning_rate / sqrt(k + sum(gparam^2)) * gparam
        ref : Chris Dyer : Notes on AdaGrad
        """
        self.lr = sharedX(learning_rate)
        self.mom = sharedX(momentum)
        self.k = sharedX(k)

    def update(self, deltas, params, gparams):
        updates = []
        for delta, param, gparam in zip(deltas, params, gparams):
            eps = theano.shared(self.k.get_value() * np.ones_like(delta.get_value(borrow=True, return_internal_type=True)))
            updates.append((eps, eps + gparam ** 2))
            updates.append((delta, self.mom * delta - self.lr * gparam / T.sqrt(eps)))
            updates.append((param, param+delta))
        return updates


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

    def update(self, deltas, params, gparams):
        updates = []
        for delta, param, gparam in zip(deltas, params, gparams):
            gparam_mean = theano.shared(np.zeros_like(delta.get_value(borrow=True, return_internal_type=True)))
            updates.append((gparam_mean, self.rho * gparam_mean + (1-self.rho) * gparam**2))
            delta_mean = theano.shared(np.zeros_like(delta.get_value(borrow=True, return_internal_type=True)))
            updates.append((delta_mean, self.rho * delta_mean + (1-self.rho) * delta**2))
            updates.append((delta, -T.sqrt(delta_mean+self.eps) / T.sqrt(gparam_mean+self.eps) * gparam))
            updates.append((param, param+delta))
        return updates


class RMSprop(LearningMethod):

    def __init__(self, learning_rate=0.01, eps=1e-6, rho=0.9):
        self.lr = sharedX(learning_rate)
        self.eps = sharedX(eps)
        self.rho = sharedX(rho)

    def update(self, deltas, params, gparams):
        updates = []
        for delta, param, gparam in zip(deltas, params, gparams):
            new_delta = self.rho * delta + (1-self.rho) * gparam**2
            new_param = param - self.lr * gparam / T.sqrt(new_delta + self.eps)
            updates.append((delta, new_delta))
            updates.append((param, new_param))
        return updates


class Adam(LearningMethod):

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.lr = sharedX(learning_rate)
        self.iter = sharedX(0)
        self.beta_1 = sharedX(beta_1)
        self.beta_2 = sharedX(beta_2)
        self.eps = sharedX(eps)

    def update(self, deltas, params, gparams):
        t = self.iter + 1
        lr_t = self.lr * T.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)
        updates = []
        for delta, param, gparam in zip(deltas, params, gparams):
            m = sharedX(param.get_value() * 0.)
            v = sharedX(param.get_value() * 0.)
            m_t = (self.beta_1 * m) + (1 - self.beta_1) * gparam
            v_t = (self.beta_2 * v) + (1 - self.beta_2) * gparam**2
            param_t = param - lr_t * m_t / (T.sqrt(v_t) + self.eps)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, param_t))
        return updates
