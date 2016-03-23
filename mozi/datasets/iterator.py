
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
        super(SequentialRecurrentIterator, self).__init__(dataset_size, batch_size)
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
        number of sequences. example of seq_len 3 and batch_size 4 will generate
        [0, 1, 2; 1, 2, 3; 2, 3, 4; 3, 4, 5]
        '''
        super(SequentialRecurrentIterator, self).__init__(dataset_size, batch_size)
        assert dataset_size >= seq_len, 'size of dataset has to be at least larger than sequence length'
        self.seq_len = seq_len
        self.ridx = np.concatenate([np.arange(self.seq_len) + i for i in range(batch_size)])

    def __iter__(self):
        self.ridx = np.concatenate([np.arange(self.seq_len) + i for i in range(batch_size)])

    def next(self):
        if self.ridx[-1] >= self.dataset_size:
            last = self.ridx[-1] - self.dataset_size + 1
            if len(self.ridx[:-last*self.seq_len]) == 0:
                raise StopIteration()
            ridx = np.copy(self.ridx)
            self.ridx += self.batch_size
            return ridx[:-last*self.seq_len]
        else:
            ridx = np.copy(self.ridx)
            self.ridx += self.batch_size
            return ridx

class SequentialRecurrentIteratorBorderMode(SubsetIterator):

    def __init__(self, dataset_size, batch_size, seq_len):
        '''
        This is for generating sequences of equal len (seq_len) with (batch_size)
        number of sequences. example of seq_len 3 and batch_size 4 will generate
        with dataset_size 4
        [0, 0, 1; 0, 1, 2;, 1, 2, 3; 2, 3, 3 ]
        '''
        super(SequentialRecurrentIterator, self).__init__(dataset_size, batch_size)
        assert dataset_size >= seq_len, 'size of dataset has to be at least larger than sequence length'
        assert seq_len % 2 == 1, 'seq_len has to be odd'
        self.seq_len = seq_len
        # self.ridx = np.concatenate([np.arange(self.seq_len) + i for i in range(batch_size)])
        self.ridx = np.arange(self.dataset_size)

    def __iter__(self):
        # self.ridx = np.concatenate([np.arange(self.seq_len) + i for i in range(batch_size)])
        self.ridx = np.arange(self.dataset_size)


    def next(self):
        if self.ridx[-1] >= self.dataset_size:
            last = self.ridx[-1] - self.dataset_size + 1
            if len(self.ridx[:-last*self.seq_len]) == 0:
                raise StopIteration()
            ridx = np.copy(self.ridx)
            self.ridx += self.batch_size
            return ridx[:-last*self.seq_len]
        else:
            ridx = np.copy(self.ridx)
            self.ridx += self.batch_size
            return ridx

    def next(self):
        buf = self.seq_len / 2

        if i-buf < 0 and i+buf > self.dataset_size-1:
            left = np.tile(0, buf-i)
            right = np.tile(self.dataset_size-1, i+buf-self.dataset_size+1)


        buf = window/2
        for i in range(len(arr_npy)):
            if i-buf < 0 and i+buf > len(arr_npy)-1:
                left = np.tile(arr_npy[0], buf-i)
                right = np.tile(arr_npy[-1], i+buf-len(arr_npy)+1)
                if np.ndim(arr_npy) == 4:
                    arr_tmp = np.rollaxis(arr_npy, 0, 4)
                    d0,d1,d2,d3 = arr_tmp.shape
                    arr_tmp = arr_tmp.reshape(d0,d1,d2*d3)
                else:
                    arr_tmp = arr_npy.flatten()
                arr = np.concatenate([left,arr_tmp,right], axis=-1)

            elif i-buf < 0:
                left = np.tile(arr_npy[0], buf-i)
                if np.ndim(arr_npy) == 4:
                    arr_tmp = np.rollaxis(arr_npy[0:i+buf+1], 0, 4)
                    d0,d1,d2,d3 = arr_tmp.shape
                    arr_tmp = arr_tmp.reshape(d0,d1,d2*d3)
                else:
                    arr_tmp = arr_npy[0:i+buf+1].flatten()
                arr = np.concatenate([left, arr_tmp], axis=-1)

            elif i+buf > len(arr_npy)-1:
                right = np.tile(arr_npy[-1], i+buf-len(arr_npy)+1)
                if np.ndim(arr_npy) == 4:
                    arr_tmp = np.rollaxis(arr_npy[i-buf:], 0, 4)
                    d0,d1,d2,d3 = arr_tmp.shape
                    arr_tmp = arr_tmp.reshape(d0,d1,d2*d3)
                else:
                    arr_tmp = arr_npy[i-buf:].flatten()
                arr = np.concatenate([arr_tmp, right], axis=-1)

            elif i >= buf and i <= len(arr_npy)-1-buf:
                if np.ndim(arr_npy) == 4:
                    arr_tmp = np.rollaxis(arr_npy[i-buf:i+buf+1], 0, 4)
                    d0,d1,d2,d3 = arr_tmp.shape
                    arr = arr_tmp.reshape(d0,d1,d2*d3)
                else:
                    arr = arr_npy[i-buf:i+buf+1].flatten()
