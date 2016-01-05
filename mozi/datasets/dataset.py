

import mozi.datasets.iterator as iterators
import numpy as np
import theano
floatX = theano.config.floatX

import logging
internal_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from mozi.log import Log

class IterMatrix(object):

    def __init__(self, X, y, iter_class='SequentialSubsetIterator',
                batch_size=100, num_batches=None, rng=None):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.iter_class = iter_class
        self.rng = rng
        self.iterator = getattr(iterators, self.iter_class)

    def __iter__(self):
        return self.iterator(dataset_size=self.dataset_size,
                            batch_size=self.batch_size,
                            num_batches=self.num_batches,
                            rng=self.rng)

    def set_iterator(self, iterator):
        self.iterator = iterator

    def __getitem__(self, key):
        return self.X[key], self.y[key]

    @property
    def dataset_size(self):
        return self.X.shape[0] if self.X is not None else -1


class IterDatasets(object):

    def __init__(self, datasets, labels, iter_class='SequentialSubsetIterator',
                batch_size=100, num_batches=None, rng=None):

        self.datasets = datasets
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.iter_class = iter_class
        self.rng = rng
        self.iterator = getattr(iterators, self.iter_class)

    def __iter__(self):
        return self.iterator(dataset_size=self.dataset_size,
                            batch_size=self.batch_size,
                            num_batches=self.num_batches,
                            rng=self.rng)

    def set_iterator(self, iterator):
        self.iterator = iterator

    def __getitem__(self, key):
        Xslice = []
        yslice = []
        for dataset in self.datasets:
            Xslice.append(dataset[key])
        for label in self.labels:
            yslice.append(label[key])
        return Xslice + yslice

    @property
    def dataset_size(self):
        return len(self.datasets[0]) if self.datasets is not None else -1


class Dataset(object):

    def __init__(self, train_valid_test_ratio=[8,1,1], log=None, batch_size=100,
                 num_batches=None, iter_class='SequentialSubsetIterator', rng=None):

        assert len(train_valid_test_ratio) == 3, 'the size of list is not 3'
        self.ratio = train_valid_test_ratio
        self.iter_class = iter_class
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = rng

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
        super(SingleBlock, self).__init__(train_valid_test_ratio, log, **kwargs)

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

    def __init__(self, data_paths, train_valid_test_ratio=[8,1,1], log=None, **kwargs):

        """
        DESCRIPTION:
            This is class for processing blocks of data, whereby dataset is loaded
            and unloaded into memory one block at a time.
        PARAM:
            data_paths(list): contains the paths to the numpy data files. It's a
                            list of tuples whereby the first element of the tuple
                            is the X path, and the second is the y path.
                            example [(X_path1, y_path1),(X_path2, y_path2)]
        """
        super(DataBlocks, self).__init__(train_valid_test_ratio, log, **kwargs)
        assert isinstance(data_paths, list), "data_paths is not a list"
        self.data_paths = data_paths
        self.single_block = SingleBlock(None, None, train_valid_test_ratio, log, **kwargs)

    def __iter__(self):
        self.files = iter(self.data_paths)
        return self

    def next(self):
        file = next(self.files)

        assert isinstance(file, tuple) or isintance(file, list), str(type(file)) + "is not a tuple or list"
        with open(file[0], 'rb') as X_fin, open(file[1], 'rb') as y_fin:
            X = np.load(X_fin)
            y = np.load(y_fin)

        self.single_block.set_Xy(X=X, y=y)
        return self.single_block

    @property
    def nblocks(self):
        return len(self.data_paths)


class MultiInputsData(SingleBlock):

    def __init__(self, datasets, labels, train_valid_test_ratio=[8,1,1], log=None, **kwargs):

        """
        DESCRIPTION:
            This class is used for multitask learning where we have multiple data
            inputs and multiple data output.
        PARAM:
            datasets (tuple of arrays or just one array of X): If our input is X1 and X2, both
            with same number of rows, then X = (X1, X2)
            labels (tuple of arrays or just one array of y): label of same number of rows as
            input data
        """

        if isinstance(datasets, tuple) or isinstance(datasets, list):
            self.num_examples = len(datasets[0])
            for dataset in datasets:
                assert len(dataset) == self.num_examples, 'number of rows for different datasets is not the same'
        else:
            self.num_examples = len(datasets)
            datasets = [datasets]

        if isinstance(labels, tuple) or isinstance(labels, list):
            for label in labels:
                assert len(label) == self.num_examples, 'number of rows for different labels is not the same'
        else:
            assert len(labels) == self.num_examples, 'number of rows for labels is not the same as input features'
            labels = [labels]

        super(MultiInputsData, self).__init__(train_valid_test_ratio, log, **kwargs)

        self.train = IterDatasets(None, None, **kwargs)
        self.valid = IterDatasets(None, None, **kwargs)
        self.test = IterDatasets(None, None, **kwargs)
        self.set(datasets, labels)


    def set(self, datasets, labels):
        total_ratio = sum(self.ratio)
        num_train = int(float(self.ratio[0]) * self.num_examples / total_ratio)
        num_valid = int(float(self.ratio[1]) * self.num_examples / total_ratio)

        trainset = []
        validset = []
        testset = []
        for dataset in datasets:
            trainset.append(dataset[:num_train])
            validset.append(dataset[num_train:num_train+num_valid])
            testset.append(dataset[num_train+num_valid:])

        trainlbl = []
        validlbl = []
        testlbl = []
        for label in labels:
            trainlbl.append(label[:num_train])
            validlbl.append(label[num_train:num_train+num_valid])
            testlbl.append(label[num_train+num_valid:])

        self.train.datasets = trainset
        self.train.labels = trainlbl

        if self.ratio[1] == 0:
            self.log.info('Valid set is empty! It is needed for early stopping and saving best model')
        self.valid.datasets = validset
        self.valid.labels = validlbl

        if self.ratio[2] == 0:
            self.log.info('Test set is empty! It is needed for testing the best model')
        self.test.datasets = testset
        self.test.labels = testlbl
