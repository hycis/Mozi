

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import theano

from pynet.datasets.dataset import SingleBlock, DataBlocks
from pynet.utils.utils import make_one_hot




class I2R_Posterior_Blocks_ClnDNN_CleanFeat(DataBlocks):


    def __init__(self, feature_size, target_size, **kwargs):

        """
        DESCRIPTION:
            This is class for processing blocks of data.
        PARAM:
            data_paths(list): contains the paths to the numpy data files. It's a
                            list of tuples whereby the first element of the tuple
                            is the X path, and the second is the y path.
        """
        dir = '/home/stuwzhz/datasets/spectral-features/npy2'
        self.data_paths = [('%s/ClnDNN_CleanFeat.post_%.3d.npy'%(dir,i), '%s/onehot_clean.pdf_%.3d.npy'%(dir,i)) for i in xrange(1)]


        super(I2R_Posterior_Blocks_ClnDNN_CleanFeat, self).__init__(feature_size=feature_size,
                                                    target_size=target_size,
                                                    data_paths=self.data_paths,
                                                    **kwargs)

class I2R_Posterior_Blocks_ClnDNN_NoisyFeat(DataBlocks):


    def __init__(self, feature_size, target_size, one_hot=True, num_blocks=30, **kwargs):

        """
        DESCRIPTION:
            This is class for processing blocks of data.
        PARAM:
            data_paths(list): contains the paths to the numpy data files. It's a
                            list of tuples whereby the first element of the tuple
                            is the X path, and the second is the y path.
        """
        dir = '/scratch/stuwzhz/dataset/npy'
        if one_hot:
            self.data_paths = [('%s/ClnDNN_NoisyFeat.post_%.3d.npy'%(dir,i), '%s/onehot_clean.pdf_%.3d.npy'%(dir,i)) for i in xrange(num_blocks)]
        else:
            self.data_paths = [('%s/ClnDNN_NoisyFeat.post_%.3d.npy'%(dir,i), '%s/clean.pdf_%.3d.npy'%(dir,i)) for i in xrange(num_blocks)]
        super(I2R_Posterior_Blocks_ClnDNN_NoisyFeat, self).__init__(feature_size=feature_size,
                                                             target_size=target_size,
                                                             data_paths=self.data_paths,
                                                             **kwargs)

class I2R_Posterior_Gaussian_Noisy_Sample(SingleBlock):

    def __init__(self, **kwargs):
        dir = '/home/stuwzhz/datasets/spectral-features/npy2'

        with open('%s/sample_y.npy'%dir) as yin:
            y = np.load(yin)
            y_tmp = []
            for e in y:
                if e > 150:
                    y_tmp.append(e)

            y_tmp = np.asarray(y_tmp)
            y_tmp = make_one_hot(y_tmp, 1998)

        super(I2R_Posterior_Gaussian_Noisy_Sample, self).__init__(X=y_tmp, y=y_tmp, **kwargs)

class I2R_Posterior_NoisyFeat_Sample(SingleBlock):

    def __init__(self, **kwargs):
        dir = '/home/stuwzhz/datasets/spectral-features/npy2'

        with open('%s/sample_y.npy'%dir) as yin, \
            open('%s/sample_X.npy'%dir) as Xin:
            y = np.load(yin)
            y = make_one_hot(y, 1998)
            X = np.load(Xin)

        super(I2R_Posterior_NoisyFeat_Sample, self).__init__(X=X, y=y, **kwargs)
