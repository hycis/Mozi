import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano
from pynet.datasets.dataset import Dataset, DataBlocks
import glob
import gc

from pynet.utils.check_memory import print_mem_usage

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

    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Blocks, self).__init__(feature_size, target_size, **kwargs)
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
        if self.dataset.preprocessor is not None:
            logger.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
            data = self.dataset.preprocessor.apply(data)
        self.dataset.set_Xy(X=data, y=data)
        data = None
        return self.dataset

    def nblocks(self):
        return len(self.parts)

class Laura_Warp_Blocks(Laura_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks, self).__init__(feature_size, target_size, **kwargs)
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

class Laura_Warp_Standardize_Blocks(Laura_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Standardize_Blocks, self).__init__(feature_size, target_size, **kwargs)
        self.parts = [ 'Laura_warp_standardize_data_000.npy',  'Laura_warp_standardize_data_010.npy',
                       'Laura_warp_standardize_data_001.npy',  'Laura_warp_standardize_data_011.npy',
                       'Laura_warp_standardize_data_002.npy',  'Laura_warp_standardize_data_012.npy',
                       'Laura_warp_standardize_data_003.npy',  'Laura_warp_standardize_data_013.npy',
                       'Laura_warp_standardize_data_004.npy',  'Laura_warp_standardize_data_014.npy',
                       'Laura_warp_standardize_data_005.npy',  'Laura_warp_standardize_data_015.npy',
                       'Laura_warp_standardize_data_006.npy',  'Laura_warp_standardize_data_016.npy',
                       'Laura_warp_standardize_data_007.npy',  'Laura_warp_standardize_data_017.npy',
                       'Laura_warp_standardize_data_008.npy',  'Laura_warp_standardize_data_018.npy',
                       'Laura_warp_standardize_data_009.npy',  'Laura_warp_standardize_data_019.npy']

        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_warp_standardize_npy'

class Laura_Standardize_Blocks(Laura_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Standardize_Blocks, self).__init__(feature_size, target_size, **kwargs)
        self.parts = [ 'Laura_standardize_data_000.npy',  'Laura_standardize_data_010.npy',
                       'Laura_standardize_data_001.npy',  'Laura_standardize_data_011.npy',
                       'Laura_standardize_data_002.npy',  'Laura_standardize_data_012.npy',
                       'Laura_standardize_data_003.npy',  'Laura_standardize_data_013.npy',
                       'Laura_standardize_data_004.npy',  'Laura_standardize_data_014.npy',
                       'Laura_standardize_data_005.npy',  'Laura_standardize_data_015.npy',
                       'Laura_standardize_data_006.npy',  'Laura_standardize_data_016.npy',
                       'Laura_standardize_data_007.npy',  'Laura_standardize_data_017.npy',
                       'Laura_standardize_data_008.npy',  'Laura_standardize_data_018.npy',
                       'Laura_standardize_data_009.npy',  'Laura_standardize_data_019.npy']

        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_standardize_npy'

class Laura_Warp_Blocks_500_RELU(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_500_RELU, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0713_Warp_500_20140714_1317_43818059'

class Laura_Warp_Blocks_500_Tanh(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_500_Tanh, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0830_Warp_Blocks_2049_500_tanh_gpu_20140902_0012_36590657'

class Laura_Warp_Blocks_180_Tanh(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_180_Tanh, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0914_Warp_Blocks_2layers_finetune_2049_180_gpu_20140915_0006_11454520'

class Laura_Warp_Blocks_650(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_650, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0721_Warp_Blocks_650_20140722_2217_09001837'

class Laura_Warp_Blocks_1000(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_1000, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0713_Warp_1000_20140714_1831_00043080'

class Laura_Warp_Blocks_250(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_250, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0717_warp_1000fea_20140717_1705_04859196'

class Laura_Warp_Blocks_180(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_180, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0721_Warp_Blocks_500_180_20140723_0131_18179134'

class Laura_Warp_Blocks_150(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Warp_Blocks_150, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0721_Warp_Blocks_180_150_20140723_1912_01578422'

class Laura_Cut_Warp_Blocks_700(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Cut_Warp_Blocks_700, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/cut_0_700_laura_warp_npy'

class Laura_Cut_Warp_Blocks_300(Laura_Warp_Blocks):
    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Cut_Warp_Blocks_300, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_warp_cut_AE0730_Cut_Warp_Blocks_700_300_20140730_0134_17129588'

class Laura_Blocks_500_Tanh_Tanh(Laura_Blocks):

    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Blocks_500_Tanh_Tanh, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0912_Blocks_2049_500_tanh_tanh_gpu_20140914_1211_46292389'

class Laura_Blocks_500(Laura_Blocks):

    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Blocks_500, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0712_500_20140713_0345_22901754'

class Laura_Blocks_1000(Laura_Blocks):

    def __init__(self, feature_size, target_size, **kwargs):
        super(Laura_Blocks_1000, self).__init__(feature_size, target_size, **kwargs)
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0712_Warp_1000_20140712_1230_54443469'
