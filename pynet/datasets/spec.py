import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano
from pynet.datasets.dataset import Dataset, DataBlocks, SingleBlock
import glob
import gc

from pynet.utils.check_memory import print_mem_usage

class P276(SingleBlock):

    def __init__(self, **kwargs):

        data_dir = os.environ['PYNET_DATA_PATH'] + '/p276'
        with open(data_dir + '/p276_data_000.npy') as f:
            data = np.load(f)

        super(P276, self).__init__(X=data, y=data, **kwargs)

class Laura_Test(SingleBlock):

    def __init__(self, **kwargs):
        data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_npy'
        with open(data_dir + '/Laura_data_000.npy') as f:
            data = np.load(f)

        super(Laura_Test, self).__init__(X=data, y=data, **kwargs)


class Laura_Root(DataBlocks):

    def __init__(self, data_paths, **kwargs):
        super(Laura_Root, self).__init__(data_paths=data_paths, **kwargs)

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





class Laura_Blocks(Laura_Root):

    def __init__(self, data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_npy', **kwargs):
        parts = [ 'Laura_data_000.npy',  'Laura_data_010.npy',
                   'Laura_data_001.npy',  'Laura_data_011.npy',
                   'Laura_data_002.npy',  'Laura_data_012.npy',
                   'Laura_data_003.npy',  'Laura_data_013.npy',
                   'Laura_data_004.npy',  'Laura_data_014.npy',
                   'Laura_data_005.npy',  'Laura_data_015.npy',
                   'Laura_data_006.npy',  'Laura_data_016.npy',
                   'Laura_data_007.npy',  'Laura_data_017.npy',
                   'Laura_data_008.npy',  'Laura_data_018.npy',
                   'Laura_data_009.npy',  'Laura_data_019.npy']
        data_paths = ["%s/%s"%(data_dir, part) for part in parts]
        super(Laura_Blocks, self).__init__(data_paths=data_paths, **kwargs)


class Laura_Warp_Blocks(Laura_Root):
    def __init__(self, data_dir=os.environ['PYNET_DATA_PATH'] + '/Laura_warp_npy', **kwargs):
        parts = [ 'Laura_warp_data_000.npy',  'Laura_warp_data_010.npy',
                   'Laura_warp_data_001.npy',  'Laura_warp_data_011.npy',
                   'Laura_warp_data_002.npy',  'Laura_warp_data_012.npy',
                   'Laura_warp_data_003.npy',  'Laura_warp_data_013.npy',
                   'Laura_warp_data_004.npy',  'Laura_warp_data_014.npy',
                   'Laura_warp_data_005.npy',  'Laura_warp_data_015.npy',
                   'Laura_warp_data_006.npy',  'Laura_warp_data_016.npy',
                   'Laura_warp_data_007.npy',  'Laura_warp_data_017.npy',
                   'Laura_warp_data_008.npy',  'Laura_warp_data_018.npy',
                   'Laura_warp_data_009.npy',  'Laura_warp_data_019.npy']
        data_paths = ["%s/%s"%(data_dir, part) for part in parts]
        super(Laura_Warp_Blocks, self).__init__(data_paths=data_paths, **kwargs)


class Laura_Warp_Standardize_Blocks(Laura_Root):
    def __init__(self, data_dir=os.environ['PYNET_DATA_PATH'] + '/Laura_warp_standardize_npy', **kwargs):
        parts = [ 'Laura_warp_standardize_data_000.npy',  'Laura_warp_standardize_data_010.npy',
                       'Laura_warp_standardize_data_001.npy',  'Laura_warp_standardize_data_011.npy',
                       'Laura_warp_standardize_data_002.npy',  'Laura_warp_standardize_data_012.npy',
                       'Laura_warp_standardize_data_003.npy',  'Laura_warp_standardize_data_013.npy',
                       'Laura_warp_standardize_data_004.npy',  'Laura_warp_standardize_data_014.npy',
                       'Laura_warp_standardize_data_005.npy',  'Laura_warp_standardize_data_015.npy',
                       'Laura_warp_standardize_data_006.npy',  'Laura_warp_standardize_data_016.npy',
                       'Laura_warp_standardize_data_007.npy',  'Laura_warp_standardize_data_017.npy',
                       'Laura_warp_standardize_data_008.npy',  'Laura_warp_standardize_data_018.npy',
                       'Laura_warp_standardize_data_009.npy',  'Laura_warp_standardize_data_019.npy']
        data_paths = ["%s/%s"%(data_dir, part) for part in parts]
        super(Laura_Warp_Standardize_Blocks, self).__init__(data_paths=data_paths, **kwargs)

class Laura_Standardize_Blocks(Laura_Root):
    def __init__(self, data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_standardize_npy', **kwargs):
        parts = [ 'Laura_standardize_data_000.npy',  'Laura_standardize_data_010.npy',
                       'Laura_standardize_data_001.npy',  'Laura_standardize_data_011.npy',
                       'Laura_standardize_data_002.npy',  'Laura_standardize_data_012.npy',
                       'Laura_standardize_data_003.npy',  'Laura_standardize_data_013.npy',
                       'Laura_standardize_data_004.npy',  'Laura_standardize_data_014.npy',
                       'Laura_standardize_data_005.npy',  'Laura_standardize_data_015.npy',
                       'Laura_standardize_data_006.npy',  'Laura_standardize_data_016.npy',
                       'Laura_standardize_data_007.npy',  'Laura_standardize_data_017.npy',
                       'Laura_standardize_data_008.npy',  'Laura_standardize_data_018.npy',
                       'Laura_standardize_data_009.npy',  'Laura_standardize_data_019.npy']
        super(Laura_Standardize_Blocks, self).__init__(data_paths=data_paths, **kwargs)

class Laura_Warp_Blocks_500_RELU(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0713_Warp_500_20140714_1317_43818059'
        super(Laura_Warp_Blocks_500_RELU, self).__init__(data_dir=data_dir, **kwargs)

class Laura_Warp_Blocks_500_Tanh(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0830_Warp_Blocks_2049_500_tanh_gpu_20140902_0012_36590657'
        super(Laura_Warp_Blocks_500_Tanh, self).__init__(data_dir=data_dir, **kwargs)

class Laura_Warp_Blocks_500_Tanh_Noisy_Clean(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura/noisy/AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_clean_continue_20141110_1235_21624029'
        super(Laura_Warp_Blocks_500_Tanh_Noisy_Clean, self).__init__(data_dir=self.data_dir, **kwargs)

class Laura_Warp_Blocks_500_Tanh_Noisy_BatchOut(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura/noisy/AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_batchout_continue_20141111_0957_22484008'
        super(Laura_Warp_Blocks_500_Tanh_Noisy_BatchOut, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_500_Tanh_Noisy_BlackOut(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura/noisy/AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_blackout_continue_20141110_1249_12963320'
        super(Laura_Warp_Blocks_500_Tanh_Noisy_BlackOut, self).__init__(data_dir=self.data_dir, **kwargs)

class Laura_Warp_Blocks_500_Tanh_Noisy_Gaussian(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura/noisy/AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_gaussian_continue_20141110_1250_49502872'
        super(Laura_Warp_Blocks_500_Tanh_Noisy_Gaussian, self).__init__(data_dir=self.data_dir, **kwargs)

class Laura_Warp_Blocks_500_Tanh_Noisy_MaskOut(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura/noisy/AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_maskout_continue_20141110_1251_56190462'
        super(Laura_Warp_Blocks_500_Tanh_Noisy_MaskOut, self).__init__(data_dir=self.data_dir, **kwargs)

class Laura_Warp_Blocks_500_Tanh_Dropout(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0916_Warp_Blocks_2049_500_tanh_tanh_gpu_dropout_20140916_1705_29139505'
        super(Laura_Warp_Blocks_500_Tanh_Dropout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_500_Tanh_Blackout(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/AE1018_Warp_Blocks_2049_500_tanh_tanh_gpu_blackout_continue_20141018_1408_44747438'
        super(Laura_Warp_Blocks_500_Tanh_Blackout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_180_Tanh_Blackout(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/AE1018_Warp_Blocks_500_180_tanh_tanh_gpu_blackout_20141018_2238_56949300'
        super(Laura_Warp_Blocks_180_Tanh_Blackout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_180_Tanh_Dropout(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0918_Warp_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy_20140918_2113_17247388'
        super(Laura_Warp_Blocks_180_Tanh_Dropout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_180_Tanh(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0914_Warp_Blocks_2layers_finetune_2049_180_gpu_20140915_0006_11454520'
        super(Laura_Warp_Blocks_180_Tanh, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Scale_Warp_Blocks_500_Tanh(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/AE0930_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140930_1345_29800576'
        super(Laura_Scale_Warp_Blocks_500_Tanh, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Scale_Warp_Blocks_180_Tanh(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE1001_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_clean_20141002_0348_53679208'
        super(Laura_Scale_Warp_Blocks_180_Tanh, self).__init__(data_dir=self.data_dir, **kwargs)



class Laura_Scale_Warp_Blocks_500_Tanh_Dropout(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/AE1002_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_dropout_20141001_0321_33382955'
        super(Laura_Scale_Warp_Blocks_500_Tanh_Dropout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Scale_Warp_Blocks_180_Tanh_Dropout(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE1001_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_dropout_20141001_2158_16765065'
        super(Laura_Scale_Warp_Blocks_180_Tanh_Dropout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_650(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0721_Warp_Blocks_650_20140722_2217_09001837'
        super(Laura_Warp_Blocks_650, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_1000(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0713_Warp_1000_20140714_1831_00043080'
        super(Laura_Warp_Blocks_1000, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_250(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0717_warp_1000fea_20140717_1705_04859196'
        super(Laura_Warp_Blocks_250, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_180(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0721_Warp_Blocks_500_180_20140723_0131_18179134'
        super(Laura_Warp_Blocks_180, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Warp_Blocks_150(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0721_Warp_Blocks_180_150_20140723_1912_01578422'
        super(Laura_Warp_Blocks_150, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Cut_Warp_Blocks_700(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/cut_0_700_laura_warp_npy'
        super(Laura_Cut_Warp_Blocks_700, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Cut_Warp_Blocks_300(Laura_Warp_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_warp_cut_AE0730_Cut_Warp_Blocks_700_300_20140730_0134_17129588'
        super(Laura_Cut_Warp_Blocks_300, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Blocks_500_Tanh_Tanh(Laura_Blocks):

    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0912_Blocks_2049_500_tanh_tanh_gpu_20140914_1211_46292389'
        super(Laura_Blocks_500_Tanh_Tanh, self).__init__(data_dir=self.data_dir, **kwargs)

class Laura_Blocks_180_Tanh_Tanh(Laura_Blocks):

    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0917_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_clean_20140917_1009_07286035'
        super(Laura_Blocks_180_Tanh_Tanh, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Blocks_500_Tanh_Tanh_Dropout(Laura_Blocks):

    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0915_Blocks_2049_500_tanh_tanh_gpu_Dropout_20140915_1900_37160748'
        super(Laura_Blocks_500_Tanh_Tanh_Dropout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Blocks_180_Tanh_Tanh_Dropout(Laura_Blocks):

    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0917_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy_20140917_1013_42539511'
        super(Laura_Blocks_180_Tanh_Tanh_Dropout, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Blocks_500_Tanh_Sigmoid(Laura_Blocks):
    def __init__(self, **kwargs):
        self.data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0915_Blocks_2049_500_tanh_sig_gpu_Dropout_20140915_1857_22433203'
        super(Laura_Blocks_500_Tanh_Sigmoid, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Blocks_500(Laura_Blocks):

    def __init__(self, **kwargs):
        data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0712_500_20140713_0345_22901754'
        super(Laura_Blocks_500, self).__init__(data_dir=self.data_dir, **kwargs)


class Laura_Blocks_1000(Laura_Blocks):

    def __init__(self, **kwargs):
        data_dir = os.environ['PYNET_DATA_PATH'] + '/Laura_AE0712_Warp_1000_20140712_1230_54443469'
        super(Laura_Blocks_1000, self).__init__(data_dir=self.data_dir, **kwargs)
