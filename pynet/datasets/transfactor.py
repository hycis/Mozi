
import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano
from pynet.datasets.dataset import Dataset, DataBlocks, SingleBlock
import glob
import gc

class TransFactor_Root(DataBlocks):

    def __init__(self, data_paths, **kwargs):
        super(TransFactor_Root, self).__init__(data_paths=data_paths, **kwargs)

    def __iter__(self):
        self.files = iter(self.data_paths)
        return self

    def next(self):
        file = next(self.files)
        with open(file, 'rb') as f:
            data = np.load(f)

        if self.preprocessor and self.noise:
            logger.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
            proc_data = self.preprocessor.apply(data)
            data = None
            logger.info('..applying noise: ' + self.noise.__class__.__name__)
            noisy_data = self.noise.apply(proc_data)

            self.single_block.set_Xy(X=noisy_data, y=proc_data)
            return self.single_block

        if self.noise:
            logger.info('..applying noise: ' + self.noise.__class__.__name__)
            noisy_data = self.noise.apply(data)

            self.single_block.set_Xy(X=noisy_data, y=data)
            return self.single_block

        if self.preprocessor:
            logger.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
            proc_data = self.preprocessor.apply(data)
            data = None
            self.single_block.set_Xy(X=proc_data, y=proc_data)
            return self.single_block

        self.single_block.set_Xy(X=data, y=data)
        return self.single_block

class TransFactor_Blocks(TransFactor_Root):

    def __init__(self, data_dir = '/RQexec/hycis/GenomeProject/dataset', **kwargs):
        parts =   [ 'X500_000small.npy', 'X500_001small.npy',
                    'X500_002small.npy', 'X500_003small.npy',
                    'X500_004small.npy', 'X500_005small.npy',
                    'X500_006small.npy', 'X500_007small.npy']
        self.data_dir = data_dir
        data_paths = ["%s/%s"%(data_dir, part) for part in parts]
        super(TransFactor_Blocks, self).__init__(data_paths=data_paths, **kwargs)

class TransFactor_Blocks150(TransFactor_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = '/RQexec/hycis/GenomeProject/dataset/AE1216_Transfactor_blocks_500_150small_20141215_1748_06646265'
        super(TransFactor_Blocks150, self).__init__(data_dir=self.data_dir, **kwargs)


class TransFactor(DataBlocks):

    def __init__(self, feature_size, target_size, **kwargs):



        self.data_dir = '/RQexec/hycis/GenomeProject/dataset'

        data_paths = [(self.data_dir + '/X500_000.npy', self.data_dir + '/y500_000_one_hot.npy'),
                            (self.data_dir + '/X500_001.npy', self.data_dir + '/y500_001_one_hot.npy'),
                            (self.data_dir + '/X500_002.npy', self.data_dir + '/y500_002_one_hot.npy')]


        super(TransFactor, self).__init__(data_paths=data_paths,
                                           feature_size=feature_size,
                                           target_size=target_size, **kwargs)

class TransFactor_AE(SingleBlock):

    def __init__(self, one_hot=True, **kwargs):

        self.data_dir = '/RQexec/hycis/GenomeProject/dataset'

        with open(self.data_dir + '/sliced40_X500_001small.npy') as Xin:
            X = np.load(Xin)
        super(TransFactor_AE, self).__init__(X=X, y=X, **kwargs)

class TransFactor_AE150(SingleBlock):

    def __init__(self, one_hot=True, **kwargs):

        self.data_dir = '/RQexec/hycis/GenomeProject/dataset/AE1215_Transfactor_500_150_20141215_0203_14098208'

        with open(self.data_dir + '/sliced40_X500_001.npy') as Xin:
            X = np.load(Xin)
        super(TransFactor_AE150, self).__init__(X=X, y=X, **kwargs)



class TransFactor_MLP(SingleBlock):

    def __init__(self, **kwargs):

        self.data_dir = '/RQexec/hycis/GenomeProject/dataset'

        with open(self.data_dir + '/sliced40_X500_001small.npy') as Xin:
            X = np.load(Xin)
        with open(self.data_dir +'/onehot_sliced40_y500_001.npy') as yin:
            y = np.load(yin)
        super(TransFactor_MLP, self).__init__(X=X, y=y, **kwargs)


class TransFactor_MLP50(SingleBlock):

    def __init__(self, **kwargs):

        self.data_dir = '/RQexec/hycis/GenomeProject/dataset'

        with open(self.data_dir + '/ae50_sliced40_X500_001.npy') as Xin:
            X = np.load(Xin)
        with open(self.data_dir +'/onehot_sliced40_y500_001.npy') as yin:
            y = np.load(yin)
        super(TransFactor_MLP50, self).__init__(X=X, y=y, **kwargs)
