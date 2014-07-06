
"""
Functionality for preprocessing Datasets. With Preprocessor, GCN, Standardize adapted from pylearn2
"""

import sys
import copy
import logging
import time
import warnings
import numpy as np
try:
    from scipy import linalg
except ImportError:
    warnings.warn("Could not import scipy.linalg")
from theano import function
import theano.tensor as T

log = logging.getLogger(__name__)

class Preprocessor(object):
    """
        Adapted from pylearn2

        Abstract class.

        An object that can preprocess a dataset.

        Preprocessing a dataset implies changing the data that
        a dataset actually stores. This can be useful to save
        memory--if you know you are always going to access only
        the same processed version of the dataset, it is better
        to process it once and discard the original.

        Preprocessors are capable of modifying many aspects of
        a dataset. For example, they can change the way that it
        converts between different formats of data. They can
        change the number of examples that a dataset stores.
        In other words, preprocessors can do a lot more than
        just example-wise transformations of the examples stored
        in the dataset.
    """

    def apply(self, X):
        """
            dataset: The dataset to act on.
            can_fit: If True, the Preprocessor can adapt internal parameters
                     based on the contents of dataset. Otherwise it must not
                     fit any parameters, or must re-use old ones.

            Typical usage:
                # Learn PCA preprocessing and apply it to the training set
                my_pca_preprocessor.apply(training_set, can_fit = True)
                # Now apply the same transformation to the test set
                my_pca_preprocessor.apply(test_set, can_fit = False)

            Note: this method must take a dataset, rather than a numpy ndarray,
                  for a variety of reasons:
                      1) Preprocessors should work on any dataset, and not all
                         datasets will store their data as ndarrays.
                      2) Preprocessors often need to change a dataset's metadata.
                         For example, suppose you have a DenseDesignMatrix dataset
                         of images. If you implement a fovea Preprocessor that
                         reduces the dimensionality of images by sampling them finely
                         near the center and coarsely with blurring at the edges,
                         then your preprocessor will need to change the way that the
                         dataset converts example vectors to images for visualization.
        """

        raise NotImplementedError(str(type(self))+" does not implement an apply method.")

    def invert(self, X):
        """
        Do any necessary prep work to be able to support the "inverse" method
        later. Default implementation is no-op.
        """
        raise NotImplementedError(str(type(self))+" does not implement an invert method.")

class ExamplewisePreprocessor(Preprocessor):
    """
        Abstract class.

        A Preprocessor that restricts the actions it can do in its
        apply method so that it could be implemented as a Block's
        perform method.

        In other words, this Preprocessor can't modify the Dataset's
        metadata, etc.

        TODO: can these things fit themselves in their apply method?
        That seems like a difference from Block.
    """

    def as_block(self):
        raise NotImplementedError(str(type(self))+" does not implement as_block.")

class Standardize(ExamplewisePreprocessor):
    """
    Adapted from pylearn2
    Subtracts the mean and divides by the standard deviation.
    """
    def __init__(self, global_mean=False, global_std=False, std_eps=1e-4, can_fit=True):
        """
        Initialize a Standardize preprocessor.

        Parameters
        ----------
        global_mean : bool
            If `True`, subtract the (scalar) mean over every element
            in the design matrix. If `False`, subtract the mean from
            each column (feature) separately. Default is `False`.
        global_std : bool
            If `True`, after centering, divide by the (scalar) standard
            deviation of every element in the design matrix. If `False`,
            divide by the column-wise (per-feature) standard deviation.
            Default is `False`.
        std_eps : float
            Stabilization factor added to the standard deviations before
            dividing, to prevent standard deviations very close to zero
            from causing the feature values to blow up too much.
            Default is `1e-4`.
        """
        self._global_mean = global_mean
        self._global_std = global_std
        self._std_eps = std_eps
        self._mean = None
        self._std = None
        self.can_fit = can_fit

    def apply(self, X):
        if self.can_fit:
            self._mean = X.mean() if self._global_mean else X.mean(axis=0)
            self._std = X.std() if self._global_std else X.std(axis=0)
        else:
            if self._mean is None or self._std is None:
                raise ValueError("can_fit is False, but Standardize object "
                                 "has no stored mean or standard deviation")
        X = (X - self._mean) / (self._std_eps + self._std)
        return X


class GCN(Preprocessor):

    """
    Adapted from pylearn2
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).

    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and
        features indexed on the second.

    scale : float, optional
        Multiply features by this const.

    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing.
        Defaults to `False`.

    use_std : bool, optional
        Normalize by the per-example standard deviation across features
        instead of the vector norm.

    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.

    min_divisor : float, optional
        If the divisor for an example is less than this value,
        do not apply it. Defaults to `1e-8`.
    """

    def __init__(self, scale=1., subtract_mean=False, use_std=False,
                sqrt_bias=0., min_divisor=1e-8):

        self.scale = scale
        self.subtract_mean = subtract_mean
        self.use_std = use_std
        self.sqrt_bias = sqrt_bias
        self.min_divisor = min_divisor

    def apply(self, X):
        """
        Returns
        -------
        Xp : ndarray, 2-dimensional
            The contrast-normalized features.

        Notes
        -----
        `sqrt_bias` = 10 and `use_std = True` (and defaults for all other
        parameters) corresponds to the preprocessing used in [1].

        .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
           Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
           http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
        """
        assert X.ndim == 2, "X.ndim must be 2"
        scale = float(self.scale)
        # Note: this is per-example mean across pixels, not the
        # per-pixel mean across examples. So it is perfectly fine
        # to subtract this without worrying about whether the current
        # object is the train, valid, or test set.
        if self.subtract_mean:
            self.mean = X.mean(axis=1)[:, np.newaxis]
            X = X - self.mean  # Makes a copy.
        else:
            X = X.copy()
        if self.use_std:
            # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
            # Coates' code does.
            self.normalizers = np.sqrt(self.sqrt_bias + X.var(axis=1, ddof=1)) / scale
        else:
            self.normalizers = np.sqrt(self.sqrt_bias + (X ** 2).sum(axis=1)) / scale
        # Don't normalize by anything too small.
        self.normalizers[self.normalizers < self.min_divisor] = 1.
        X /= self.normalizers[:, np.newaxis]  # Does not make a copy.
        return X

    def invert(self, X):
        try:
            if self.subtract_mean:
                X = X + self.mean
            rval = X * self.normalizers[:, np.newaxis]
            return rval
        except AttributeError:
            print 'apply() needs to be used before invert()'
        except:
            print "Unexpected error:", sys.exc_info()[0]

class LogGCN(GCN):

    def __init__(self, positive_values=True, **kwarg):
        '''
        postive_values: bool
            indicates whether the output of the processor should be scaled to be positive
        '''
        self.positive_values = positive_values;
        super(LogGCN, self).__init__(**kwarg)

    def apply(self, X):
        if self.positive_values:
            rval = X + 1
        rval = np.log(rval)
        return super(LogGCN, self).apply(rval)

    def invert(self, X):
        X = super(LogGCN, self).invert(X)
        if self.positive_values:
            return np.exp(X) - 1
        else:
            return np.exp(X)






class Scale(Preprocessor):

    """
    Scale the input into a range

    Parameters
    ----------
    X : ndarray, 2-dimensional
        numpy matrix with examples indexed on the first axis and
        features indexed on the second.

    global_max : real
        the maximum value of the whole dataset. If not provided, global_max is set to X.max()

    global_min : real
        the minimum value of the whole dataset. If not provided, global_min is set to X.min()

    scale_range : size 2 list
        set the upper bound and lower bound after scaling

    buffer : float
        the buffer on the upper lower bound such that [L+buffer, U-buffer]
    """


    def __init__(self, global_max=None, global_min=None, scale_range=[0,1], buffer=1e-8):

        self.scale_range = scale_range
        self.buffer = buffer
        self.max = global_max
        self.min = global_min
        assert scale_range[0] + buffer < scale_range[1] - buffer, \
                'the lower bound is larger than the upper bound'

    def apply(self, X):

        self.max = self.max if self.max is not None else X.max()
        self.min = self.min if self.min is not None else X.min()
        width = self.max - self.min
        assert width > 0, 'the max is not bigger than the min'
        scale = (self.scale_range[1] - self.scale_range[0] - 2 * self.buffer) / width
        X = scale * (X - self.min)
        X = X + self.scale_range[0] + self.buffer

        return X

    def invert(self, X):
        if self.max is None or self.min is None:
            raise ValueError('to use invert, either global_max and global_min are provided or \
                                apply(X) is used before')
        width = self.max - self.min
        assert width > 0, 'the max is not bigger than the min'
        scale = width / (self.scale_range[1] - self.scale_range[0] - 2 * self.buffer)
        X = scale * (X - self.scale_range[0] - self.buffer)
        X = X + self.min

        return X
