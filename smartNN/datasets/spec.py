


import os
import numpy as np
import theano
from smartNN.datasets.dataset import IterMatrix, Dataset
import glob
import numpy as np

class P276(Dataset):
    
    def __init__(self, feature_size=2049, train_ratio=5, 
                    valid_ratio=1, test_ratio=1, **kwargs):
        
#         data_dir = '/RQusagers/hycis/smartNN/data/p276'
        data_dir = os.environ['smartNN_DATA_PATH'] + '/p276'
        with open(data_dir + '/p276.npy') as f:
            data = np.load(f)
            
        size = data.shape[0]
        assert size % feature_size == 0, 'feature size is not a multiple of data size'
        
        num_examples = size / feature_size
        data = data.reshape(num_examples, feature_size)
        total_ratio = train_ratio + valid_ratio + test_ratio
        num_train = int(train_ratio * 1.0 * num_examples / total_ratio)
        num_valid = int((valid_ratio + train_ratio) * 1.0 * num_examples / total_ratio)
                
        train_set = data[:num_train]
        valid_set = data[num_train:num_valid]
        test_set = data[num_valid:]

        super(P276, self).__init__(train=[train_set, train_set], 
                                    valid=[valid_set, valid_set], 
                                    test=[test_set, test_set], **kwargs)

     