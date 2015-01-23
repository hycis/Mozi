import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano
from pynet.datasets.dataset import Dataset, DataBlocks
import glob

class Laura_Blocks_GCN_Mapping(DataBlocks):

    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Blocks_GCN_Mapping, self).__init__(feature_size, target_size, **kwargs)
        self.parts = [ 'Laura_data_000.npy',  'Laura_data_010.npy',
                       'Laura_data_001.npy',  'Laura_data_011.npy',
                       'Laura_data_002.npy',  'Laura_data_012.npy',
                       'Laura_data_003.npy',  'Laura_data_013.npy',
                       'Laura_data_004.npy',  'Laura_data_014.npy',
                       'Laura_data_005.npy',  'Laura_data_015.npy',
                       'Laura_data_006.npy',  'Laura_data_016.npy',
                       'Laura_data_007.npy',  'Laura_data_017.npy',
                       'Laura_data_008.npy',  'Laura_data_018.npy',
                       'Laura_data_009.npy',  'Laura_data_019.npy']

        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_npy'


    def __iter__(self):
        self.files = iter(self.parts)
        return self

    def next(self):
        self.dataset.train = None
        self.dataset.valid = None
        self.dataset.test = None
        with open(self.data_dir + '/' + next(self.files), 'rb') as f:
            data = np.load(f)
        assert self.dataset.preprocessor is not None \
                and self.dataset.preprocessor.__class__.__name__ == 'GCN'
        logger.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
        data = self.dataset.preprocessor.apply(data)
        normal = self.dataset.preprocessor.normalizers
        self.dataset.set_Xy(X=data, y=normal.reshape((normal.shape[0], 1)))
        data = None
        return self.dataset

    def nblocks(self):
        return len(self.parts)

class Laura_Warp_Blocks_GCN_Mapping(Laura_Blocks_GCN_Mapping):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_GCN_Mapping, self).__init__(feature_size, target_size, **kwargs)
        self.parts = [ 'Laura_warp_data_000.npy',  'Laura_warp_data_010.npy',
                       'Laura_warp_data_001.npy',  'Laura_warp_data_011.npy',
                       'Laura_warp_data_002.npy',  'Laura_warp_data_012.npy',
                       'Laura_warp_data_003.npy',  'Laura_warp_data_013.npy',
                       'Laura_warp_data_004.npy',  'Laura_warp_data_014.npy',
                       'Laura_warp_data_005.npy',  'Laura_warp_data_015.npy',
                       'Laura_warp_data_006.npy',  'Laura_warp_data_016.npy',
                       'Laura_warp_data_007.npy',  'Laura_warp_data_017.npy',
                       'Laura_warp_data_008.npy',  'Laura_warp_data_018.npy',
                       'Laura_warp_data_009.npy',  'Laura_warp_data_019.npy']

        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_warp_npy'
