


import os
os.environ['PYNET_DATA_PATH'] = '/Volumes/Storage/Dropbox/CodingProjects/pynet/data/'

from pynet.datasets.mnist import Mnist
from pynet.datasets.preprocessor import GCN

data = Mnist()
train = data.get_train()
proc = GCN()
out = proc.apply(train.X)
inv = proc.invert(out)