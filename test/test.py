


import os
os.environ['MOZI_DATA_PATH'] = '/Volumes/Storage/Dropbox/CodingProjects/mozi/data/'

from mozi.datasets.mnist import Mnist
from mozi.datasets.preprocessor import GCN

data = Mnist()
train = data.get_train()
proc = GCN()
out = proc.apply(train.X)
inv = proc.invert(out)
