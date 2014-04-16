
# -----------------------------------[ Model Description ]----------------------------------
# -- SmartNN is a conditional wide neural net, whereby each layer is divided into segments. 
# -- Each segment represents an expert, and for each expert, it has a mirrored gater. Each gater is a
# -- single neuron. All the mirrored gaters for all the experts are connected to form a mirrored gating net.
# -- All the units in the gating net are softmax.
# 
# -- [ Training Algorithm ]
# -- 1. Pass the example through the gating net, all the units
# 
# 


import os

import theano
import theano.tensor as T
import numpy as np

NNdir = os.path.dirname(os.path.realpath(__file__))
NNdir = os.path.dirname(NNdir)


if not os.getenv('smartNN_DATA_PATH'):
    os.environ['smartNN_DATA_PATH'] = NNdir + '/data'

if not os.getenv('smartNN_DATABASE_PATH'):
    os.environ['smartNN_DATABASE_PATH'] = NNdir + '/database'

if not os.getenv('smartNN_SAVE_PATH'):
    os.environ['smartNN_SAVE_PATH'] = NNdir + '/save'


print('smartNN_DATA_PATH = ' + os.environ['smartNN_DATA_PATH'])
print('smartNN_SAVE_PATH = ' + os.environ['smartNN_SAVE_PATH'])
print('smartNN_DATABASE_PATH = ' + os.environ['smartNN_DATABASE_PATH'])

from smartNN.mlp import MLP
from smartNN.layer import RELU, Sigmoid, Softmax, Linear
from smartNN.datasets.mnist import Mnist
from smartNN.learning_rule import LearningRule
from smartNN.log import Log
from smartNN.train_object import TrainObject
from smartNN.cost import Cost
from smartNN.datasets.preprocessor import Standardize, GCN, Scale

from smartNN.datasets.spec import P276
    


def savenpy(folder_path):
    import glob
    import itertools
    import matplotlib.pyplot as plt
    
    files = glob.glob(folder_path + '/*.spec.double')
    size = len(files)
    data = []
    count = 0
    for f in files:
        with open(f) as fb:
            print('..file: ', f)
            clip = np.fromfile(fb, dtype='<f4', count=-1)
            data.extend(clip)
#             plt.plot(clip[50:2049])
#             plt.show()
#             import pdb
#             pdb.set_trace()
        print(str(count) + '/' + str(size) + '..done '  + f)
        
        count += 1

    with open(folder_path + '/p276.npy', 'wb') as f:
        np.save(f, data)

    print('all finished successfully')
    
def test():
    from smartNN.utils.database_utils import display_database 
    
    display_database(os.environ['smartNN_DATABASE_PATH'] + '/Database_Name.db', 'testing')

def unpickle_mlp(model):
    import cPickle
    from pylearn2.utils.image import tile_raster_images
    from PIL.Image import fromarray
    from smartNN.datasets.preprocessor import GCN, Standardize
    
    with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + model + '/model.pkl', 'rb') as f:
        mlp = cPickle.load(f)
    
    data = Mnist(preprocessor = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator',
                    rng = None)
    
    test = data.get_test()
#     prep = Standardize()
    prep = GCN(use_std = False)
    test.X = prep.apply(test.X)
    
    orig_array = tile_raster_images(X = test.X[0:100], img_shape=(28,28), tile_shape=(10,10), 
                                    tile_spacing=(5, 5), scale_rows_to_unit_interval=True, output_pixel_vals=True)
    orig_im = fromarray(orig_array)
    orig_im.save(NNdir + '/images/' + model + '_orig.jpeg')
    print('orig image saved. Opening image..')
    orig_im.show()
    
    new_X = mlp.fprop(test.X)
    new_array = tile_raster_images(X = new_X[0:100], img_shape=(28,28), tile_shape=(10,10), 
                                    tile_spacing=(0, 0), scale_rows_to_unit_interval=True, output_pixel_vals=True)
    new_im = fromarray(new_array)
    import pdb
    pdb.set_trace()
    new_im.save(NNdir + '/images/' + model + '_reconstruct.jpeg')
    print('reconstruct image saved. Opening image..') 
    new_im.show()

def plot_spec(model):
    
    from smartNN.datasets.preprocessor import GCN, Scale
    
    with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + model + '/model.pkl', 'rb') as f:
        mlp = cPickle.load(f)
    
    data = P276(feature_size=2049, train_ratio=8, 
                valid_ratio=1, test_ratio=1)
    
    test = data.get_test()
    test.X = prep.apply(test.X)
    
    

def test_AE():

    import cPickle
    
    AE1 = 'stacked_AE3_layer1_20140407_0142_53816454'
    AE2 = 'stacked_AE3_layer2_20140407_0144_52735085'
    model = 'stacked_AE_layer3_20140407_0019_48317469'
    
    data = Mnist(preprocessor = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator',
                    rng = None)
                    
    with open(os.environ['smartNN_SAVE_PATH'] + '/' + AE1 + '/model.pkl', 'rb') as f:

        mlp1 = cPickle.load(f)
    
    mlp1.pop_layer(-1)
    reduced_test_X = mlp1.fprop(data.get_test().X)
    
    with open(os.environ['smartNN_SAVE_PATH'] + '/' + AE2 + '/model.pkl', 'rb') as f:
        mlp2 = cPickle.load(f)
    
    output = mlp2.fprop(reduced_test_X)
    import pdb
    pdb.set_trace()

    

if __name__ == '__main__':
#     spec()
#     savenpy('/Applications/VCTK/data/inter-module/mcep/England/p276')
#     savenpy('/home/zhenzhou/VCTK/Research-Demo/fa-tts/STRAIGHT-TTS/tmp/England/p276')
#     test()
#     unpickle_mlp('stacked_AE5_20140410_0707_50423148')
#     test_AE()
                                
                                
                                
                         