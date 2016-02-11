
"""
Functionality for preprocessing Datasets. With Preprocessor, GCN, Standardize adapted from pylearn2
"""

import sys
import copy
import logging
import time
import warnings
import numpy as np
import sklearn.preprocessing as preproc
try:
    from scipy import linalg
except ImportError:
    warnings.warn("Could not import scipy.linalg")
from theano import function
import theano.tensor as T
import theano
import scipy

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
    def __init__(self, global_mean=None, global_std=None, std_eps=1e-4):
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
        self._std_eps = std_eps
        self._mean = global_mean
        self._std = global_std

    def apply(self, X):
        if self._mean is None:
            self._mean = X.mean(axis=0)
        if self._std is None:
            self._std = X.std(axis=0)
        X = (X - self._mean) / (self._std_eps + self._std)
        return X

    def invert(self, X):
        return X * (self._std_eps + self._std) + self._mean






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

    def __init__(self, scale=1., subtract_mean=True, use_std=False,
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
            self.mean = np.mean(X, axis=1)
            X = X - self.mean[:, np.newaxis]  # Makes a copy.
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
        X = X / self.normalizers[:, np.newaxis]  # Does not make a copy.
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


class GCN_IMG(GCN):

    def apply(self, X):
        assert X.ndim == 4, 'img dimension should be 4 of (b, c, h, w)'
        b, c, h, w = X.shape
        newX = super(GCN_IMG, self).apply(X.reshape((b*c, h*w)))
        return newX.reshape((b,c,h,w))

    def invert(self, X):
        assert X.ndim == 4, 'img dimension should be 4 of (b, c, h, w)'
        b, c, h, w = X.shape
        newX = super(GCN_IMG, self).invert(X.reshape((b*c, h*w)))
        return newX.reshape((b,c,h,w))


class LogGCN(GCN):

    def __init__(self, positive_values=True, **kwarg):
        '''
        postive_values: bool
            indicates whether the output of the processor should be scaled to be positive
        '''
        self.positive_values = positive_values
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

class Log(Preprocessor):

    def __init__(self, positive_values=False, **kwarg):
        '''
        postive_values: bool
            indicates whether the output of the processor should be scaled to be positive
        '''
        self.positive_values = positive_values

    def apply(self, X):
        if self.positive_values:
            X = X + 1
        return np.log(X)

    def invert(self, X):
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


    def __init__(self, global_max=None, global_min=None, scale_range=[-1,1], buffer=0.1):

        self.scale_range = scale_range
        self.buffer = buffer
        self.max = global_max
        self.min = global_min
        assert scale_range[0] + buffer < scale_range[1] - buffer, \
                'the lower bound is larger than the upper bound'

    def apply(self, X):
        self.max = self.max if self.max else X.max()
        self.min = self.min if self.min else X.min()
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


class Pipeline(Preprocessor):

    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def apply(self, X):
        newX = X.copy()
        for proc in self.preprocessors:
            newX = proc.apply(newX)
        return newX

    def invert(self, X):
        newX = X.copy()
        for proc in self.preprocessors:
            newX = proc.invert(newX)
        return newX

class Normalize(Preprocessor):

    def __init__(self, norm='l2', axis=1, channelwise=False):
        """
        normalize each data vector to unit length

        Parameters
        ----------
        X : ndarray, 2-dimensional
            numpy matrix with examples indexed on the first axis and
            features indexed on the second.
        norm : l1, l2 or max
        channelwise: apply preprocessing channelwise
        """
        self.norm = norm
        self.axis = axis
        self.channelwise = channelwise


    def apply(self, X):
        if X.ndim == 4 and self.channelwise:
            shape = X.shape
            flattern_X = np.reshape(X, (shape[0]*shape[1], shape[2]*shape[3]))
            flattern_X = preproc.normalize(flattern_X, norm=self.norm, axis=1, copy=True)
            return flattern_X.reshape(shape)

        if X.ndim > 2:
            shape = X.shape
            flattern_X = np.reshape(X, (shape[0], np.prod(shape[1:])))
            flattern_X = preproc.normalize(flattern_X, norm=self.norm, axis=self.axis, copy=True)
            return flattern_X.reshape(shape)

        return preproc.normalize(X, norm=self.norm, axis=self.axis, copy=True)


class Sigmoid(Preprocessor):

    def apply(self, X):
        return 1 / (1 + np.exp(-X))

    def invert(self, X):
        return np.log(X / (1-X + 1e-9))


class ZCA(Preprocessor):

    """
    from pylearn2
    Performs ZCA whitening.
    .. TODO::
        WRITEME properly
        add reference
    Parameters
    ----------
    n_components : integer, optional
        Keeps the n_components biggest eigenvalues and corresponding
        eigenvectors of covariance matrix.
    n_drop_components : integer, optional
        Drops the n_drop_components smallest eigenvalues and corresponding
        eigenvectors of covariance matrix. Will only drop components
        when n_components is not set i.e. n_components has preference over
        n_drop_components.
    filter_bias : float, optional
        TODO: verify that default of 0.1 is what was used in the
        Coates and Ng paper, add reference
    store_inverse : bool, optional
        When self.apply(dataset, can_fit=True) store not just the
        preprocessing matrix, but its inverse. This is necessary when
        using this preprocessor to instantiate a ZCA_Dataset.
    """

    def __init__(self, n_components=None, n_drop_components=None,
                 filter_bias=0.1, store_inverse=True):
        warnings.warn("This ZCA preprocessor class is known to yield very "
                      "different results on different platforms. If you plan "
                      "to conduct experiments with this preprocessing on "
                      "multiple machines, it is probably a good idea to do "
                      "the preprocessing on a single machine and copy the "
                      "preprocessed datasets to the others, rather than "
                      "preprocessing the data independently in each "
                      "location.")
        # TODO: test to see if differences across platforms
        # e.g., preprocessing STL-10 patches in LISA lab versus on
        # Ian's Ubuntu 11.04 machine
        # are due to the problem having a bad condition number or due to
        # different version numbers of scipy or something
        self.n_components = n_components
        self.n_drop_components = n_drop_components
        self.copy = True
        self.filter_bias = np.cast[theano.config.floatX](filter_bias)
        self.has_fit_ = False
        self.store_inverse = store_inverse
        self.P_ = None  # set by fit()
        self.inv_P_ = None  # set by fit(), if self.store_inverse is True

        # Analogous to DenseDesignMatrix.design_loc. If not None, the
        # matrices P_ and inv_P_ will be saved together in <save_path>
        # (or <save_path>.npz, if the suffix is omitted).
        self.matrices_save_path = None

    @staticmethod
    def _gpu_matrix_dot(matrix_a, matrix_b, matrix_c=None):
        """
        Performs matrix multiplication.
        Attempts to use the GPU if it's available. If the matrix multiplication
        is too big to fit on the GPU, this falls back to the CPU after throwing
        a warning.
        Parameters
        ----------
        matrix_a : WRITEME
        matrix_b : WRITEME
        matrix_c : WRITEME
        """
        if not hasattr(ZCA._gpu_matrix_dot, 'theano_func'):
            ma, mb = T.matrices('A', 'B')
            mc = T.dot(ma, mb)
            ZCA._gpu_matrix_dot.theano_func = \
                theano.function([ma, mb], mc, allow_input_downcast=True)

        theano_func = ZCA._gpu_matrix_dot.theano_func

        try:
            if matrix_c is None:
                return theano_func(matrix_a, matrix_b)
            else:
                matrix_c[...] = theano_func(matrix_a, matrix_b)
                return matrix_c
        except MemoryError:
            warnings.warn('Matrix multiplication too big to fit on GPU. '
                          'Re-doing with CPU. Consider using '
                          'THEANO_FLAGS="device=cpu" for your next '
                          'preprocessor run')
            return np.dot(matrix_a, matrix_b, matrix_c)

    @staticmethod
    def _gpu_mdmt(mat, diags):
        """
        Performs the matrix multiplication M * D * M^T.
        First tries to do this on the GPU. If this throws a MemoryError, it
        falls back to the CPU, with a warning message.
        Parameters
        ----------
        mat : WRITEME
        diags : WRITEME
        """

        floatX = theano.config.floatX

        # compile theano function
        if not hasattr(ZCA._gpu_mdmt, 'theano_func'):
            t_mat = T.matrix('M')
            t_diags = T.vector('D')
            result = T.dot(t_mat * t_diags, t_mat.T)
            ZCA._gpu_mdmt.theano_func = theano.function(
                [t_mat, t_diags],
                result,
                allow_input_downcast=True)

        try:
            # function()-call above had to downcast the data. Emit warnings.
            if str(mat.dtype) != floatX:
                warnings.warn('Implicitly converting mat from dtype=%s to '
                              '%s for gpu' % (mat.dtype, floatX))
            if str(diags.dtype) != floatX:
                warnings.warn('Implicitly converting diag from dtype=%s to '
                              '%s for gpu' % (diags.dtype, floatX))

            return ZCA._gpu_mdmt.theano_func(mat, diags)

        except MemoryError:
            # fall back to cpu
            warnings.warn('M * D * M^T was too big to fit on GPU. '
                          'Re-doing with CPU. Consider using '
                          'THEANO_FLAGS="device=cpu" for your next '
                          'preprocessor run')
            return np.dot(mat * diags, mat.T)

    def set_matrices_save_path(self, matrices_save_path):
        """
        Analogous to DenseDesignMatrix.use_design_loc().
        If a matrices_save_path is set, when this ZCA is pickled, the internal
        parameter matrices will be saved separately to `matrices_save_path`, as
        a numpy .npz archive. This uses half the memory that a normal pickling
        does.
        Parameters
        ----------
        matrices_save_path : WRITEME
        """
        if matrices_save_path is not None:
            assert isinstance(matrices_save_path, str)
            matrices_save_path = os.path.abspath(matrices_save_path)

            if os.path.isdir(matrices_save_path):
                raise IOError('Matrix save path "%s" must not be an existing '
                              'directory.')

            assert matrices_save_path[-1] not in ('/', '\\')
            if not os.path.isdir(os.path.split(matrices_save_path)[0]):
                raise IOError('Couldn\'t find parent directory:\n'
                              '\t"%s"\n'
                              '\t of matrix path\n'
                              '\t"%s"')

        self.matrices_save_path = matrices_save_path

    def __getstate__(self):
        """
        Used by pickle.  Returns a dictionary to pickle in place of
        self.__dict__.
        If self.matrices_save_path is set, this saves the matrices P_ and
        inv_P_ separately in matrices_save_path as a .npz archive, which uses
        much less space & memory than letting pickle handle them.
        """
        result = copy.copy(self.__dict__)  # shallow copy
        if self.matrices_save_path is not None:
            matrices = {'P_': self.P_}
            if self.inv_P_ is not None:
                matrices['inv_P_'] = self.inv_P_

            np.savez(self.matrices_save_path, **matrices)

            # Removes the matrices from the dictionary to be pickled.
            for key, matrix in matrices.items():
                del result[key]

        return result

    def __setstate__(self, state):
        """
        Used to unpickle.
        Parameters
        ----------
        state : dict
            The dictionary created by __setstate__, presumably unpickled
            from disk.
        """

        # Patch old pickle files
        if 'matrices_save_path' not in state:
            state['matrices_save_path'] = None

        if state['matrices_save_path'] is not None:
            matrices = np.load(state['matrices_save_path'])

            # puts matrices' items into state, overriding any colliding keys in
            # state.
            state = dict(state.items() + matrices.items())
            del matrices

        self.__dict__.update(state)

        if not hasattr(self, "inv_P_"):
            self.inv_P_ = None

    def fit(self, X):
        """
        Fits this `ZCA` instance to a design matrix `X`.
        Parameters
        ----------
        X : ndarray
            A matrix where each row is a datum.
        Notes
        -----
        Implementation details:
        Stores result as `self.P_`.
        If self.store_inverse is true, this also computes `self.inv_P_`.
        """

        assert X.dtype in ['float32', 'float64']
        assert not np.any(np.isnan(X))
        assert len(X.shape) == 2
        n_samples = X.shape[0]
        if self.copy:
            X = X.copy()
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        log.info('computing zca of a {0} matrix'.format(X.shape))
        t1 = time.time()

        bias = self.filter_bias * scipy.sparse.identity(X.shape[1],
                                                        theano.config.floatX)

        covariance = ZCA._gpu_matrix_dot(X.T, X) / X.shape[0] + bias
        t2 = time.time()
        log.info("cov estimate took {0} seconds".format(t2 - t1))

        t1 = time.time()
        eigs, eigv = linalg.eigh(covariance)
        t2 = time.time()

        log.info("eigh() took {0} seconds".format(t2 - t1))
        assert not np.any(np.isnan(eigs))
        assert not np.any(np.isnan(eigv))
        assert eigs.min() > 0

        if self.n_components and self.n_drop_components:
            raise ValueError('Either n_components or n_drop_components'
                             'should be specified')

        if self.n_components:
            eigs = eigs[-self.n_components:]
            eigv = eigv[:, -self.n_components:]

        if self.n_drop_components:
            eigs = eigs[self.n_drop_components:]
            eigv = eigv[:, self.n_drop_components:]

        t1 = time.time()

        sqrt_eigs = np.sqrt(eigs)
        try:
            self.P_ = ZCA._gpu_mdmt(eigv, 1.0 / sqrt_eigs)
        except MemoryError:
            warnings.warn()
            self.P_ = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)

        t2 = time.time()
        assert not np.any(np.isnan(self.P_))
        self.has_fit_ = True

        if self.store_inverse:
            self.inv_P_ = ZCA._gpu_mdmt(eigv, sqrt_eigs)
        else:
            self.inv_P_ = None

    def apply(self, X, can_fit=True):
        """
        .. todo::
            WRITEME
        """
        # Compiles apply.x_minus_mean_times_p(), a numeric Theano function that
        # evauates dot(X - mean, P)
        if not hasattr(ZCA, '_x_minus_mean_times_p'):
            x_symbol = T.matrix('X')
            mean_symbol = T.vector('mean')
            p_symbol = T.matrix('P_')
            new_x_symbol = T.dot(x_symbol - mean_symbol, p_symbol)
            ZCA._x_minus_mean_times_p = theano.function([x_symbol,
                                                         mean_symbol,
                                                         p_symbol],
                                                        new_x_symbol)

        assert X.dtype in ['float32', 'float64']
        if not self.has_fit_:
            assert can_fit
            self.fit(X)

        new_X = ZCA._gpu_matrix_dot(X - self.mean_, self.P_)
        return new_X

    def invert(self, X):
        """
        .. todo::
            WRITEME
        """
        assert X.ndim == 2

        if self.inv_P_ is None:
            warnings.warn("inv_P_ was None. Computing "
                          "inverse of P_ now. This will take "
                          "some time. For efficiency, it is recommended that "
                          "in the future you compute the inverse in ZCA.fit() "
                          "instead, by passing it store_inverse=True.")
            log.info('inverting...')
            self.inv_P_ = np.linalg.inv(self.P_)
            log.info('...done inverting')

        return self._gpu_matrix_dot(X, self.inv_P_) + self.mean_
