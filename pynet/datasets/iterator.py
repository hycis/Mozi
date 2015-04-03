__author__ = "Zhenzhou Wu"
__copyright__ = "Copyright 2012, Zhenzhou Wu"
__credits__ = ["Zhenzhou Wu"]
__license__ = "3-clause BSD"
__email__ = "hyciswu@gmail.com"
__maintainer__ = "Zhenzhou Wu"

"""
Adapted from pylearn2 reference http://deeplearning.net/software/pylearn2/

Iterators providing indices for different kinds of iteration over
datasets.

Presets:
    sequential: iterates through fixed slices of the dataset in sequence
    shuffled_sequential: iterates through a shuffled version of the dataset
                 in sequence
    random_slice: on each call to next, returns a slice of the dataset,
                  chosen uniformly at random over contiguous slices
                  samples with replacement, but still reports that
                  container is empty after num_examples / batch_size calls
    random_uniform: on each call to next, returns a random subset of the
                  dataset.
                  samples with replacement, but still reports that
                  container is empty after num_examples / batch_size calls
"""
from __future__ import division
import warnings
import numpy
np = numpy
from theano import config


class SubsetIterator(object):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        """
            rng: either a seed value for a numpy RandomState or
            numpy RandomState workalike
        """
        raise NotImplementedError()

    def next(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    # Class-level attributes that might hint the behaviour of
    # FiniteDatasetIterator.

    # Does this return subsets that need fancy indexing? (i.e. lists
    # of indices)
    fancy = False

    # Does this class make use of random number generators?
    stochastic = False

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def num_examples(self):
        return self.batch_size * self.num_batches

    @property
    def uneven(self):
        return False

class SequentialSubsetIterator(SubsetIterator):
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        if rng is not None:
            raise ValueError("non-None rng argument not supported for "
                             "sequential batch iteration")
        assert num_batches is None or num_batches >= 0
        self._dataset_size = dataset_size
        if batch_size is None:
            if num_batches is not None:
                batch_size = int(numpy.ceil(self._dataset_size / num_batches))
            else:
                raise ValueError("need one of batch_size, num_batches "
                                 "for sequential batch iteration")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = numpy.ceil(self._dataset_size / batch_size)
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (self._dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = numpy.ceil(self._dataset_size / batch_size)
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._next_batch_no = 0
        self._idx = 0
        self._batch = 0
        self._indices = np.arange(self._dataset_size)


    def next(self):
        if self._batch >= self.num_batches or self._idx >= self._dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self._idx + self._batch_size) > self._dataset_size:
            self._last = self._indices[self._idx : self._dataset_size]
            self._idx = self._dataset_size
            return self._last

        else:
            self._last = self._indices[self._idx : self._idx + self._batch_size]
            self._idx += self._batch_size
            self._batch += 1
            return self._last

    fancy = False
    stochastic = False

    @property
    def num_examples(self):
        product = self.batch_size * self.num_batches
        return min(product, self._dataset_size)

    @property
    def uneven(self):
        return self.batch_size * self.num_batches > self._dataset_size


class ShuffledSequentialSubsetIterator(SequentialSubsetIterator):

    stochastic = True
    fancy = True

    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        super(ShuffledSequentialSubsetIterator, self).__init__(
            dataset_size,
            batch_size,
            num_batches,
            None
        )
        if rng is not None and hasattr(rng, 'random_integers'):
            self._rng = rng
        else:
            self._rng = numpy.random.RandomState(rng)
        self._shuffled = numpy.arange(self._dataset_size)
        self._rng.shuffle(self._shuffled)

    def next(self):
        if self._batch >= self.num_batches or self._idx >= self._dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self._idx + self._batch_size) > self._dataset_size:
            rval = self._shuffled[self._idx: self._dataset_size]
            self._idx = self._dataset_size
            return rval
        else:
            rval = self._shuffled[self._idx: self._idx + self._batch_size]
            self._idx += self._batch_size
            self._batch += 1
            return rval
