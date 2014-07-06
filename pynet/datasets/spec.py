import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano
from pynet.datasets.dataset import Dataset, DataBlocks
import glob

class P276(Dataset):

    def __init__(self, **kwargs):

        data_dir = os.environ['PYNET_DATA_PATH'] + '/p276'
        with open(data_dir + '/p276_data_000.npy') as f:
            data = np.load(f)

        super(P276, self).__init__(X=data, y=data, **kwargs)

class Laura_Test(Dataset):

    def __init__(self, **kwargs):
        data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_npy'
        with open(data_dir + '/Laura_data_000.npy') as f:
            data = np.load(f)

        super(Laura_Test, self).__init__(X=data, y=data, **kwargs)

class Laura_Blocks(DataBlocks):

    def __init__(self, feature_size, target_size, slice=[0,-1], **kwargs):

        self.parts = ['Laura_data_000.npy','Laura_data_010.npy','Laura_data_020.npy','Laura_data_030.npy',
                        'Laura_data_001.npy','Laura_data_011.npy','Laura_data_021.npy','Laura_data_031.npy',
                        'Laura_data_002.npy','Laura_data_012.npy','Laura_data_022.npy','Laura_data_032.npy',
                        'Laura_data_003.npy','Laura_data_013.npy','Laura_data_023.npy','Laura_data_033.npy',
                        'Laura_data_004.npy','Laura_data_014.npy','Laura_data_024.npy','Laura_data_034.npy',
                        'Laura_data_005.npy','Laura_data_015.npy','Laura_data_025.npy','Laura_data_035.npy',
                        'Laura_data_006.npy','Laura_data_016.npy','Laura_data_026.npy','Laura_data_036.npy',
                        'Laura_data_007.npy','Laura_data_017.npy','Laura_data_027.npy','Laura_data_037.npy',
                        'Laura_data_008.npy','Laura_data_018.npy','Laura_data_028.npy','Laura_data_038.npy',
                        'Laura_data_009.npy','Laura_data_019.npy','Laura_data_029.npy','Laura_data_039.npy']

        assert(len(slice) == 2)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_npy'
        self.slice = slice
        super(Laura_Blocks, self).__init__(feature_size, target_size, **kwargs)


    def __iter__(self):
        self.files = iter(self.parts)
        return self

    def next(self):
        with open(self.data_dir + '/' + next(self.files)) as f:
            data = np.load(f)
        if self.dataset.preprocessor is not None:
            logger.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
            data = self.dataset.preprocessor.apply(data)
        self.dataset.set_Xy(X=data, y=data)
        # [:, self.slice[0]:self.slice[1]],
        return self.dataset

    def nblocks(self):
        return len(self.parts)
