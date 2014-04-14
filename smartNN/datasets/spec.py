


import os
import numpy as np
import theano
from smartNN.datasets.dataset import IterMatrix, Dataset
import glob
import numpy as np

class P276(Dataset):
    
    def __init__(self, feature_size=2049, **kwargs):
        
#         data_dir = '/RQusagers/hycis/smartNN/data/p276'
        data_dir = os.environ['smartNN_DATA_PATH'] + '/p276'
        with open(data_dir + '/p276.npy') as f:
            data = np.load(f)
            
        size = data.shape[0]
        assert size % feature_size == 0, 'feature size is not a multiple of data size'
        
        num_examples = size / feature_size
        data = data.reshape(num_examples, feature_size)

        super(P276, self).__init__(X=data, y=data, **kwargs)

     