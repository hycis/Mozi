
import pynet.datasets.iterator as iterators
import numpy as np

import logging
internal_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from pynet.log import Log

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
        return self.iterator(dataset_size=self.dataset_size(),
                            batch_size=self.batch_size,
                            num_batches=self.num_batches,
                            rng=self.rng)

    def set_iterator(self, iterator):
        self.iterator = iterator

    def dataset_size(self):
        return self.X.shape[0]

    def feature_size(self):
        return self.X.shape[1]

    def target_size(self):
        return self.y.shape[1]


class Dataset(object):

    def __init__(self, train_valid_test_ratio = [8,1,1],
                preprocessor=None, noise=None, batch_size=100, num_batches=None,
                iter_class='SequentialSubsetIterator', rng=None, log=None):

        '''
        DESCRIPTION: Abstract class
        '''

        assert len(train_valid_test_ratio) == 3, 'the size of list is not 3'
        self.ratio = train_valid_test_ratio
        self.preprocessor = preprocessor
        self.noise = noise
        self.iter_class = iter_class
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = rng

        self.log = log

        if self.log is None:
            # use default Log setting
            self.log = Log(logger=internal_logger)

    def __iter__(self):
        raise NotImplementedError(str(type(self))+" does not implement the __iter__ method.")

    def next(self):
        raise NotImplementedError(str(type(self))+" does not implement the __iter__ method.")

    def nblocks(self):
        raise NotImplementedError(str(type(self))+" does not implement the __iter__ method.")

    def feature_size(self):
        raise NotImplementedError(str(type(self))+" does not implement the __iter__ method.")

    def target_size(self):
        raise NotImplementedError(str(type(self))+" does not implement the __iter__ method.")


class SingleBlock(Dataset):

    def __init__(self, X=None, y=None, **kwargs):
        super(SingleBlock, self).__init__(**kwargs)

        self.train = None
        self.valid = None
        self.test = None

        assert len(self.ratio) == 3, 'the size of list is not 3'

        if X and y:
            assert X.shape[0] == y.shape[0], 'the number of examples in input and target dont match'
            if self.preprocessor:
                self.log.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
                X = self.preprocessor.apply(X)
            if self.noise:
                self.log.info('..applying noise: ' + self.noise.__class__.__name__)
                X = self.noise.apply(X)

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

    def nblocks(self):
        return 1

    def feature_size(self):
        return self.train.X.shape[1]

    def target_size(self):
        return self.train.y.shape[1]

    def set_Xy(self, X, y):
        num_examples = X.shape[0]
        total_ratio = sum(self.ratio)
        num_train = int(self.ratio[0] * 1.0 * num_examples / total_ratio)
        num_valid = int(self.ratio[1] * 1.0 * num_examples / total_ratio)

        train_X = X[:num_train]
        train_y = y[:num_train]

        valid_X = X[num_train:num_train+num_valid]
        valid_y = y[num_train:num_train+num_valid]

        test_X = X[num_train+num_valid:]
        test_y = y[num_train+num_valid:]

        if self.ratio[0] == 0:
            self.log.warning('Train set is empty!')

        self.train = IterMatrix(train_X, train_y, iter_class=self.iter_class,
                                    batch_size=self.batch_size,
                                    num_batches=self.num_batches, rng=self.rng)

        if self.ratio[1] == 0:
            self.log.warning('Valid set is empty! It is needed for stopping of training')

        self.valid = IterMatrix(valid_X, valid_y, iter_class=self.iter_class,
                                    batch_size=self.batch_size,
                                    num_batches=self.num_batches, rng=self.rng)

        if self.ratio[2] == 0:
            self.log.warning('Test set is empty! It is needed for saving the best model')

        self.test = IterMatrix(test_X, test_y, iter_class=self.iter_class,
                                    batch_size=self.batch_size,
                                    num_batches=self.num_batches, rng=self.rng)


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

    def __init__(self, feature_size, target_size, data_paths, **kwargs):

        """
        DESCRIPTION:
            This is class for processing blocks of data.
        PARAM:
            data_paths(list): contains the paths to the numpy data files. It's a
                            list of tuples whereby the first element of the tuple
                            is the X path, and the second is the y path.
        """
        super(DataBlocks, self).__init__(**kwargs)
        self.featureSize = feature_size
        self.targetSize = target_size
        assert isinstance(data_paths, list), "data_paths is not a list"
        self.data_paths = data_paths
        self.single_block = SingleBlock(X=None, y=None, **kwargs)

    def __iter__(self):
        self.files = iter(self.data_paths)
        return self

    def next(self):
        file = next(self.files)
        # if isinstance(file, str):
        #     with open(file, 'rb') as f:
        #         data = np.load(f)
        #         X = data
        #
        #     if self.preprocessor:
        #         self.log.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
        #         data = self.preprocessor.apply(data)
        #         X = data
        #
        #     if self.noise:
        #         self.log.info('..applying noise: ' + self.noise.__class__.__name__)
        #         noisy_data = self.noise.apply(data)
        #         X = noisy_data
        #
        #     self.single_block.set_Xy(X=X, y=data)
        #     return self.single_block

        assert isinstance(file, tuple), str(type(file)) + "is not a tuple"
        with open(file[0], 'rb') as X_fin, open(file[1], 'rb') as y_fin:
            data = np.load(X_fin)
            X = data
            y = np.load(y_fin)

        if self.preprocessor:
            self.log.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
            data = self.preprocessor.apply(data)
            X = data

        if self.noise:
            self.log.info('..applying noise: ' + self.noise.__class__.__name__)
            noisy_data = self.noise.apply(data)
            X = noisy_data

        self.single_block.set_Xy(X=X, y=y)
        return self.single_block




    def nblocks(self):
        return len(self.data_paths)

    def feature_size(self):
        return self.featureSize

    def target_size(self):
        return self.targetSize




 #####################################
# class SingleBlock(object):
#     '''
#     Interface for Dataset with single block
#     '''
#
#     def __iter__(self):
#         self.iter = True
#         return self
#
#     def next(self):
#         if self.iter:
#             self.iter = False
#             return self
#         else:
#             raise StopIteration
#
#     def nblocks(self):
#         return 1
#
#
# class SingleBlock(Dataset):
#
#     def __init__(self, X=None, y=None, train_valid_test_ratio = [8,1,1],
#                 preprocessor=None, noise=None, batch_size=100, num_batches=None,
#                 iter_class='SequentialSubsetIterator', rng=None):
#
#         '''
#         DESCRIPTION: Interface that contains three IterMatrix
#         PARAM:
#             X : 2d numpy of size [num_examples, num_features]
#             y : 2d numpy of size [num_examples, num_targets]
#             train_valid_test_ratio : list
#                 the ratio to split the dataset linearly
#         '''
#
#         assert len(train_valid_test_ratio) == 3, 'the size of list is not 3'
#         self.ratio = train_valid_test_ratio
#         self.preprocessor = preprocessor
#         self.noise = noise
#         self.iter_class = iter_class
#         self.batch_size = batch_size
#         self.num_batches = num_batches
#         self.rng = rng
#
#         self.train = None
#         self.valid = None
#         self.test = None
#
#         assert len(train_valid_test_ratio) == 3, 'the size of list is not 3'
#
#         if X is not None and y is not None:
#             assert X.shape[0] == y.shape[0], 'the number of examples in input and target dont match'
#             if self.preprocessor:
#                 self.log.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
#                 X = self.preprocessor.apply(X)
#             if self.noise:
#                 self.log.info('..applying noise: ' + self.noise.__class__.__name__)
#                 X = self.noise.apply(X)
#
#             self.set_Xy(X, y)
#
#
#     def set_Xy(self, X, y):
#         num_examples = X.shape[0]
#         total_ratio = sum(self.ratio)
#         num_train = int(self.ratio[0] * 1.0 * num_examples / total_ratio)
#         num_valid = int(self.ratio[1] * 1.0 * num_examples / total_ratio)
#
#         train_X = X[:num_train]
#         train_y = y[:num_train]
#
#         valid_X = X[num_train:num_train+num_valid]
#         valid_y = y[num_train:num_train+num_valid]
#
#         test_X = X[num_train+num_valid:]
#         test_y = y[num_train+num_valid:]
#
#         if self.ratio[0] == 0:
#             self.log.warning('Train set is empty!')
#
#         self.train = IterMatrix(train_X, train_y, iter_class=self.iter_class,
#                                     batch_size=self.batch_size,
#                                     num_batches=self.num_batches, rng=self.rng)
#
#         if self.ratio[1] == 0:
#             self.log.warning('Valid set is empty! It is needed for stopping of training')
#
#         self.valid = IterMatrix(valid_X, valid_y, iter_class=self.iter_class,
#                                     batch_size=self.batch_size,
#                                     num_batches=self.num_batches, rng=self.rng)
#
#         if self.ratio[2] == 0:
#             self.log.warning('Test set is empty! It is needed for saving the best model')
#
#         self.test = IterMatrix(test_X, test_y, iter_class=self.iter_class,
#                                     batch_size=self.batch_size,
#                                     num_batches=self.num_batches, rng=self.rng)
#
#
#     def get_train(self):
#         return self.train
#
#     def get_valid(self):
#         return self.valid
#
#     def get_test(self):
#         return self.test
#
#     def set_train(self, X, y):
#         self.train.X = X
#         self.train.y = y
#
#     def set_valid(self, X, y):
#         self.valid.X = X
#         self.valid.y = y
#
#     def set_test(self, X, y):
#         self.test.X = X
#         self.test.y = y
#
#     def feature_size(self):
#         return self.train.X.shape[1]
#
#     def target_size(self):
#         return self.train.y.shape[1]
#
#
# class DataBlocks(Dataset):
#     '''
#     Interface for large dataset broken up into many blocks and train one block
#     of Dataset at a time
#     '''
#
#     # def __init__(self, feature_size, target_size, train_valid_test_ratio = [8,1,1],
#     #             preprocessor=None, noise=None, batch_size=100, num_batches=None,
#     #             iter_class='SequentialSubsetIterator', rng=None):
#     #
#     #     self.featureSize = feature_size
#     #     self.targetSize = target_size
#     #
#     #     self.ratio = train_valid_test_ratio
#     #     self.preprocessor = preprocessor
#     #     self.noise = noise
#     #     self.iter_class = iter_class
#     #     self.batch_size = batch_size
#     #     self.num_batches = num_batches
#     #     self.rng = rng
#     #
#     #     assert len(train_valid_test_ratio) == 3, 'the size of list is not 3'
#     #     self.dataset = Dataset(X=None, y=None,
#     #                             train_valid_test_ratio=train_valid_test_ratio,
#     #                             preprocessor=preprocessor, noise=noise, batch_size=batch_size,
#     #                             num_batches=num_batches, iter_class=iter_class, rng=rng)
#
#     def __init__(feature_size, target_size, **kwargs):
#         self.feature_size = feature_size
#         self.target_size = target_size
#         super(DataBlocks, self).__init__(**kwargs)
#
#     def __iter__(self):
#         raise NotImplementedError(str(type(self))+" does not implement the __iter__ method.")
#
#     def next(self):
#         '''
#         Loads the next Dataset() from the disk
#         '''
#         raise NotImplementedError(str(type(self))+" does not implement the next method.")
#
#     def nblocks(self):
#         raise NotImplementedError(str(type(self))+" does not implement the nblocks method.")
#
#     def feature_size(self):
#         return self.featureSize
#
#     def target_size(self):
#         return self.targetSize
