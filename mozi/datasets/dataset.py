

import mozi.datasets.iterator as iterators
import numpy as np
import theano
from multiprocessing import Process, Queue
import time
floatX = theano.config.floatX

import logging
internal_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from mozi.log import Log

class IterMatrix(object):

    def __init__(self, X, y, iter_class='SequentialSubsetIterator',
                batch_size=100, **kwargs):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.iter_class = iter_class
        self.kwargs = kwargs
        self.iterator = getattr(iterators, self.iter_class)

    def __iter__(self):
        return self.iterator(dataset_size=self.dataset_size,
                            batch_size=self.batch_size, **self.kwargs)

    def set_iterator(self, iterator):
        self.iterator = iterator

    def __getitem__(self, key):
        return self.X[key], self.y[key]

    @property
    def dataset_size(self):
        return self.X.shape[0] if self.X is not None else -1


class IterDatasets(IterMatrix):

    def __init__(self, X, y, iter_class='SequentialSubsetIterator',
                batch_size=100, **kwargs):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.iter_class = iter_class
        self.kwargs = kwargs
        self.iterator = getattr(iterators, self.iter_class)

    def __getitem__(self, key):
        Xslice = []
        yslice = []
        for dataset in self.X:
            Xslice.append(dataset[key])
        for label in self.y:
            yslice.append(label[key])
        return Xslice + yslice

    @property
    def dataset_size(self):
        if isinstance(self.X, (list, tuple)):
            dsize = len(self.y[0])
        elif X is None:
            dsize = -1
        else:
            dsize = len(self.X)
        return dsize


class Dataset(object):

    def __init__(self, log):
        self.log = log
        if self.log is None:
            # use default Log setting, using the internal logger
            self.log = Log(logger=internal_logger)


    def __iter__(self):
        raise NotImplementedError(str(type(self))+" does not implement the __iter__ method.")

    def next(self):
        raise NotImplementedError(str(type(self))+" does not implement the next method.")

    @property
    def nblocks(self):
        raise NotImplementedError(str(type(self))+" does not implement the nblocks method.")


class SingleBlock(Dataset):

    def __init__(self, X=None, y=None, train_valid_test_ratio=[8,1,1], log=None, **kwargs):
        '''
        All the data is loaded into memory for one go training
        '''
        super(SingleBlock, self).__init__(log)
        self.ratio = train_valid_test_ratio
        self.train = IterMatrix(X=None, y=None, **kwargs)
        self.valid = IterMatrix(X=None, y=None, **kwargs)
        self.test = IterMatrix(X=None, y=None, **kwargs)

        assert len(self.ratio) == 3, 'the size of list is not 3'

        if X is not None and y is not None:
            self.set_Xy(X, y)

    def __iter__(self):
        self.iter = True
        return self

    def next(self):
        if self.iter:
            # only one iteration since there is only one data block
            self.iter = False
            return self
        else:
            raise StopIteration

    @property
    def nblocks(self):
        return 1

    def set_Xy(self, X, y):
        num_examples = len(X)
        total_ratio = sum(self.ratio)
        num_train = int(self.ratio[0] * 1.0 * num_examples / total_ratio)
        num_valid = int(self.ratio[1] * 1.0 * num_examples / total_ratio)

        train_X = X[:num_train]
        train_y = y[:num_train]

        valid_X = X[num_train:num_train+num_valid]
        valid_y = y[num_train:num_train+num_valid]

        test_X = X[num_train+num_valid:]
        test_y = y[num_train+num_valid:]

        self.train.X = train_X
        self.train.y = train_y

        if self.ratio[1] == 0:
            self.log.info('Valid set is empty! It is needed for early stopping and saving best model')

        self.valid.X = valid_X
        self.valid.y = valid_y

        if self.ratio[2] == 0:
            self.log.info('Test set is empty! It is needed for testing the best model')

        self.test.X = test_X
        self.test.y = test_y


    def get_train(self):
        return self.train

    def get_valid(self):
        return self.valid

    def get_test(self):
        return self.test

    def set_train(self, X, y):
        self.train.X = X
        self.train.y = y

    def set_valid(self, X, y):
        self.valid.X = X
        self.valid.y = y

    def set_test(self, X, y):
        self.test.X = X
        self.test.y = y


class DataBlocks(Dataset):

    def __init__(self, data_paths, train_valid_test_ratio=[8,1,1], log=None, allow_preload=True, **kwargs):

        """
        DESCRIPTION:
            This is class for processing blocks of data, whereby dataset is loaded
            and unloaded into memory one block at a time.
        PARAM:
            data_paths(list): contains the paths to the numpy data files. It's a
                            list of tuples whereby the first element of the tuple
                            is the X path, and the second is the y path.
                            example [(X_path1, y_path1),(X_path2, y_path2)]
            allow_preload(bool): by allowing preload, it will preload the next data block
                            while training at the same time on the current datablock,
                            this will reduce time but will also cost more memory.

        """
        super(DataBlocks, self).__init__(train_valid_test_ratio, log, **kwargs)
        assert isinstance(data_paths, (list,tuple)), "data_paths is not a list"
        self.data_paths = data_paths
        self.single_block = SingleBlock(None, None, train_valid_test_ratio, log, **kwargs)
        self.allow_preload = allow_preload
        self.q = Queue()

    def __iter__(self):
        self.files = iter(self.data_paths)
        if self.allow_preload:
            self.lastblock = False
            bufile = next(self.files)
            self.load_Xy(bufile, self.q)
        return self

    def next(self):
        if self.allow_preload:
            if self.lastblock:
                raise StopIteration

            try:
                X, y = self.q.get(block=True, timeout=None)
                self.single_block.set_Xy(X,y)
                bufile = next(self.files)
                p = Process(target=self.load_Xy, args=(bufile, self.q))
                p.start()
            except:
                self.lastblock = True
        else:
            fpaths = next(self.files)
            X,y = self.openfile(fpaths)
            self.single_block.set_Xy(X=X, y=y)

        return self.single_block

    @staticmethod
    def openfile(paths):
        assert isinstance(paths, (list,tuple)), str(type(paths)) + "is not a tuple or list"
        with open(paths[0], 'rb') as X_fin, open(paths[1], 'rb') as y_fin:
            X = np.load(X_fin)
            y = np.load(y_fin)
        return X,y

    def load_Xy(self, paths, q):
        self.log.info('..loading: ' + paths)
        X,y = self.openfile(paths)
        self.log.info('..loaded: ' + paths)
        q.put((X,y))

    @property
    def nblocks(self):
        return len(self.data_paths)


class MultiInputsData(SingleBlock):

    def __init__(self, X=None, y=None, train_valid_test_ratio=[8,1,1], log=None, **kwargs):

        """
        DESCRIPTION:
            This class is used for multitask learning where we have multiple data
            inputs and multiple data output.
        PARAM:
            X (tuple of arrays or just one array of X): If our input is X1 and X2, both
            with same number of rows, then X = (X1, X2)
            y (tuple of arrays or just one array of y): label of same number of rows as
            input data
        """
        super(MultiInputsData, self).__init__(train_valid_test_ratio=train_valid_test_ratio,
                                              log=log, **kwargs)

        self.train = IterDatasets(None, None, **kwargs)
        self.valid = IterDatasets(None, None, **kwargs)
        self.test = IterDatasets(None, None, **kwargs)
        self.set(X, y)


    def set(self, X, y):
        if isinstance(X, (list,tuple)):
            self.num_examples = len(X[0])
            for dataset in X:
                assert len(dataset) == self.num_examples, 'number of rows for different datasets is not the same'
        elif X is None:
            self.num_examples = 0
            X = []
        else:
            self.num_examples = len(X)
            X = [X]

        if isinstance(y, (list,tuple)):
            for label in y:
                assert len(label) == self.num_examples, 'number of rows for different y is not the same'
        elif y is None:
            y = []
            assert self.num_examples == 0
        else:
            assert len(y) == self.num_examples, 'number of rows for y is not the same as input features'
            y = [y]

        total_ratio = sum(self.ratio)
        num_train = int(float(self.ratio[0]) * self.num_examples / total_ratio)
        num_valid = int(float(self.ratio[1]) * self.num_examples / total_ratio)

        trainset = []
        validset = []
        testset = []
        for dataset in X:
            trainset.append(dataset[:num_train])
            validset.append(dataset[num_train:num_train+num_valid])
            testset.append(dataset[num_train+num_valid:])

        trainlbl = []
        validlbl = []
        testlbl = []
        for label in y:
            trainlbl.append(label[:num_train])
            validlbl.append(label[num_train:num_train+num_valid])
            testlbl.append(label[num_train+num_valid:])

        self.train.X = trainset
        self.train.y = trainlbl

        if self.ratio[1] == 0:
            self.log.info('Valid set is empty! It is needed for early stopping and saving best model')
        self.valid.X = validset
        self.valid.y = validlbl

        if self.ratio[2] == 0:
            self.log.info('Test set is empty! It is needed for testing the best model')
        self.test.X = testset
        self.test.y = testlbl
