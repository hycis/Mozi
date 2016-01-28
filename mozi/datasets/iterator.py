
# from __future__ import division
import warnings
import numpy
np = numpy
from theano import config


class SubsetIterator(object):
    def __init__(self, dataset_size, batch_size=64, num_batches=None, rng=None):
        """
            rng: either a seed value for a numpy RandomState or
            numpy RandomState workalike
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = rng
        self.idx = 0

    def next(self):
        raise NotImplementedError()

    def __iter__(self):
        self.idx = 0
        return self

    # Class-level attributes that might hint the behaviour of
    # FiniteDatasetIterator.

    # Does this return subsets that need fancy indexing? (i.e. lists
    # of indices)
    fancy = False

    # Does this class make use of random number generators?
    stochastic = False

    @property
    def num_examples(self):
        return self.batch_size * self.num_batches

    @property
    def uneven(self):
        return False


class SequentialSubsetIterator(SubsetIterator):
    def __init__(self, dataset_size, batch_size, num_batches=None, rng=None):
        if rng is not None:
            raise ValueError("non-None rng argument not supported for "
                             "sequential batch iteration")
        assert num_batches is None or num_batches >= 0
        if batch_size is None:
            if num_batches is not None:
                batch_size = int(numpy.ceil(dataset_size / num_batches))
            else:
                raise ValueError("need one of batch_size, num_batches "
                                 "for sequential batch iteration")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = numpy.ceil(dataset_size / batch_size)
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = numpy.ceil(dataset_size / float(batch_size))
        self.next_batch_no = 0
        self.batch = 0
        super(SequentialSubsetIterator, self).__init__(dataset_size, batch_size, num_batches)
        self.idx = 0
        self.indices = np.arange(self.dataset_size)


    def next(self):
        if self.batch >= self.num_batches or self.idx >= self.dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self.idx + self.batch_size) > self.dataset_size:
            self.last = self.indices[self.idx : self.dataset_size]
            self.idx = self.dataset_size
            return self.last

        else:
            self.last = self.indices[self.idx : self.idx + self.batch_size]
            self.idx += self.batch_size
            self.batch += 1
            return self.last

    fancy = False
    stochastic = False

    @property
    def num_examples(self):
        product = self.batch_size * self.num_batches
        return min(product, self.dataset_size)

    @property
    def uneven(self):
        return self.batch_size * self.num_batches > self.dataset_size


class ShuffledSequentialSubsetIterator(SequentialSubsetIterator):

    stochastic = True
    fancy = True

    def __init__(self, dataset_size, batch_size, num_batches=None, rng=None):
        super(ShuffledSequentialSubsetIterator, self).__init__(
            dataset_size,
            batch_size,
            num_batches,
            None
        )
        self.idx = 0
        self.indices = np.arange(self.dataset_size)
        if rng is not None and hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)
        self.shuffled = numpy.arange(self.dataset_size)
        self.rng.shuffle(self.shuffled)

    def next(self):
        if self.batch >= self.num_batches or self.idx >= self.dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self.idx + self.batch_size) > self.dataset_size:
            rval = self.shuffled[self.idx: self.dataset_size]
            self.idx = self.dataset_size
            return rval
        else:
            rval = self.shuffled[self.idx: self.idx + self.batch_size]
            self.idx += self.batch_size
            self.batch += 1
            return rval


class SequentialContinuousIterator(SubsetIterator):

    def __init__(self, dataset_size, batch_size, step_size=1):
        '''
        The is for continous sequence with fix step at a time.
        '''
        super(SequentialRecurrentIterator, self).__init__(datast_size, batch_size)
        self.idx = 0
        self.indices = np.arange(self.dataset_size)
        self.step_size = step_size
        assert self.step_size > 0

    def next(self):
        if self.idx + self.batch_size > self.dataset_size:
            raise StopIteration()

        rval = self.indices[self.idx:self.idx+self.batch_size]
        self.idx += self.step_size
        return rval


class SequentialRecurrentIterator(SubsetIterator):

    def __init__(self, dataset_size, batch_size, seq_len):
        '''
        This is for generating sequences of equal len (seq_len) with (batch_size)
        number of sequences. example of seq_len 2 and batch_size 3 will generate
        [0, 1, 1, 2, 2, 3]
        '''
        super(SequentialRecurrentIterator, self).__init__(dataset_size, batch_size)
        assert dataset_size >= seq_len, 'size of dataset has to be at least larger than sequence length'
        self.seq_len = seq_len
        self.ridx = np.concatenate([np.arange(seq_len) + i for i in range(batch_size)])

    def __iter__(self):
        self.ridx = np.concatenate([np.arange(seq_len) + i for i in range(batch_size)])

    def next(self):
        if self.ridx[-1] >= self.dataset_size:
            last = self.ridx[-1] - self.dataset_size + 1
            if len(self.ridx[:-last*self.seq_len]) == 0:
                raise StopIteration()
            ridx = np.copy(self.ridx)
            print 'ridx[0],ridx[-1], len(ridx)', ridx[0], ridx[-1], len(ridx)
            print 'dataset size', self.dataset_size
            print
            self.ridx += self.batch_size
            return ridx[:-last*self.seq_len]
        else:
            ridx = np.copy(self.ridx)
            self.ridx += self.batch_size
            return ridx
