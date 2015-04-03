__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"


"""
Functionality : Define the noise that is to be added to the dataset
"""

import numpy as np

class Noise(object):
    """
    This is an abstract class for applying noise to dataset
    """

    def apply(self, X):
        """
        DESCRIPTION:
            This method applies noise to X and return a noisy X
        PARAM:
            X : 2d numpy array of dimension number of examples by number of dimensions
        """
        raise NotImplementedError(str(type(self))+" does not implement an apply method.")

    def invert(self, X):
        """
        DESCRIPTION:
            Remove the noise from X
        PARAM:
            X : 2d numpy array of dimension number of examples by number of dimensions
        """
        raise NotImplementedError(str(type(self))+" does not implement an invert method.")




class MaskOut(Noise):

    """
    This noise masked out a portion of the dimension from each example
    """

    def __init__(self, ratio=0.5):
        """
        PARAM:
            ratio : float
                The portion of the inputs that is masked out
        """
        self.ratio = ratio

    def apply(self, X):
        self.noise = np.random.binomial(size=X.shape, n=1, p=(1-self.ratio))
        return X * self.noise

    def invert(self, X):
        return X / self.noise


class Gaussian(Noise):
    """
    Applies gaussian noise to each value of X
    """

    def __init__(self, std=0.01, mean=0):
        self.std = std
        self.mean = mean

    def apply(self, X):
        return X + np.random.normal(loc=self.mean, scale=self.std, size=X.shape)



class BlackOut(Noise):
    """
    This noise masked out a random example in a dataset,
    adding noise in the time dimension
    """

    def __init__(self, ratio=0.5):
        """
        PARAM:
            ratio : float
                The portion of the examples that is masked out
        """
        self.ratio = ratio

    def apply(self, X):
        return X * np.random.binomial(size=(X.shape[0],1), n=1, p=(1-self.ratio))
