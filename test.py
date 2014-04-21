


import os
os.environ['smartNN_DATA_PATH'] = '/Volumes/Storage/Dropbox/CodingProjects/smartNN/data/'

from smartNN.datasets.mnist import Mnist
from smartNN.datasets.preprocessor import GCN

data = Mnist()
train = data.get_train()
proc = GCN()
out = proc.apply(train.X)
inv = proc.invert(out)