import os
import numpy as np
import theano
from smartNN.datasets.dataset import IterMatrix, Dataset
import glob
import numpy as np

class P276(Dataset):
    
    def __init__(self, **kwargs):
        
        data_dir = os.environ['smartNN_DATA_PATH'] + '/p276'
        with open(data_dir + '/p276_data_000.npy') as f:
            data = np.load(f)

        super(P276, self).__init__(X=data, y=data, **kwargs)


class P276_LogWarp(Dataset):

    def __init__(self, **kwargs):
        
        data_dir = os.environ['smartNN_DATA_PATH'] + '/p276'
        with open(data_dir + '/p276_data_logWarp.npy') as f:
            data = np.load(f)

        super(P276_LogWarp, self).__init__(X=data, y=data, **kwargs)

class Laura(Dataset):
    
    def __init__(self, part, **kwargs):
        
        data_path = os.environ['smartNN_DATA_PATH'] + '/Laura/' + part
        with open(data_path) as f:
            data = np.load(f)
        
        super(Laura_Splits, self).__init__(X=data, y=data, **kwargs)
        
    
    
    
  