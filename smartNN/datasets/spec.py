


import os
import numpy as np
import theano
from smartNN.datasets.dataset import IterMatrix, Dataset
import glob
import numpy as np

class P276_Spec(Dataset):
    
    def __init__(self, preprocess=None, feature_size=2049,
                batch_size=100, num_batches=None, 
                train_ratio=5, test_ratio=1,
                iter_class='ShuffledSequentialSubsetIterator'):
        
        im_dir = os.environ['smartNN_DATA_PATH'] + '/inter-module/mcep/England/p276'
        
#         files = glob.glob(im_dir + '/p276.npy')
#         
#         size = 0
#         data = np.asarray([], dtype='<f4')
#         for f in files:
#             clip = np.fromfile(f, dtype='<f4', count=-1)
#             data = np.concatenate([data, clip])
#             size += clip.shape[0]
        
        data = np.load(im_dir + '/p276.npy')
        size = data.shape[0]
        assert size % feature_size == 0, 'feature size is not a multiple of size'
        
#         import pdb
#         pdb.set_trace()
        
        data = data.reshape(size/feature_size, feature_size)
        split = int(1.0 * train_ratio / (train_ratio + test_ratio) * (size/feature_size))
#         import pdb
#         pdb.set_trace()
        train = data[:split]
        test = data[split:]
        
        

        self.train = IterMatrix(train, train, iter_class, batch_size, num_batches)
        self.valid = None
        self.test = IterMatrix(test, test, iter_class, batch_size)

        super(P276_Spec, self).__init__(train=self.train, valid=self.valid, test=self.test)
        
     