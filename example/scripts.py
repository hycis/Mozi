
import numpy as np
import glob
import itertools

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
from smartNN.datasets.preprocessor import Standardize, GCN

from smartNN.datasets.spec import P276


def savenpy(folder_path):

    
    files = glob.glob(folder_path + '/*.spec')
    size = len(files)
    data = []
    count = 0
    for f in files:
        with open(f) as fb:
            clip = np.fromfile(fb, dtype='<f4', count=-1)
            data.extend(clip)

        print(str(count) + '/' + str(size) + '..done '  + f)
        
        count += 1

    with open(folder_path + '/Laura.npy', 'wb') as f:
        np.save(f, data)

    print('all finished successfully')

def processnpy(filepath):
    
    with np.load(filepath, 'r') as obj:

        import pdb
        pdb.set_trace()
        
def testpreproc(path):
    from smartNN.datasets.preprocessor import Scale
    import numpy as np
    pre = Scale()
    with open(path) as f:
        X = np.load(f)
    X = pre.apply(X)
    import pdb
    pdb.set_trace()
    
    
def plot_compare_spec_results(model): 
    import cPickle
    from smartNN.utils.utils import tile_raster_graphs
    from PIL.Image import fromarray
    from smartNN.datasets.preprocessor import GCN, Standardize, Scale
    
    with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + model + '/model.pkl', 'rb') as f:
        mlp = cPickle.load(f)
    
#     data = Mnist(preprocessor = None, 
#                     binarize = False,
#                     batch_size = 100,
#                     num_batches = None, 
#                     train_ratio = 5, 
#                     valid_ratio = 1,
#                     iter_class = 'SequentialSubsetIterator',
#                     rng = None)

    print('..loading data from ' + model)
    data = P276(feature_size=2049, train_valid_test_ratio=[8,1,1])
    
    test = data.get_test()
#     prep = Standardize()
#     prep = Scale()
    prep = GCN()
    print('..preprocessing data ' + prep.__class__.__name__ )
    proc_test_X = prep.apply(test.X)
    print('..fprop X')
    output = mlp.fprop(proc_test_X)
    print('..saving data')
    plt = tile_raster_graphs(proc_test_X[133:134], output[133:134], slice=(0,-1),
                            tile_shape=(2,1), tile_spacing=(0.1,0.1), legend=True)
    
    plt.savefig(os.environ['smartNN_SAVE_PATH'] + '/images/' + model + '.png')
    print('Saved Successfully')
#     plt.show()

def save_AE_output(model, preproc):
    import cPickle
    from PIL.Image import fromarray
    import smartNN.datasets.preprocessor as proc
    
#     with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + model + '/model.pkl', 'rb') as f:
#         mlp = cPickle.load(f)
    
    prep = getattr(proc, preproc)()
    
#     print('..loading data from ' + model)
#     data = P276(feature_size=2049, train_valid_test_ratio=[1,0,0])
    data = Mnist(train_valid_test_ratio=[1,0,0])
    train = data.get_train()


    
    
    

if __name__ == '__main__':

#     savenpy('/RQusagers/hycis/smartNN/data/Laura')
#     testpreproc('/Applications/VCTK/data/inter-module/mcep/England/p276/p276.npy')
#     testpreproc('/data/lisa/exp/wuzhen/smartNN/data/p276/p276.npy')
#     processnpy('/Applications/VCTK/data/inter-module/mcep/England/p276/p276.npy') 
#     plot_compare_spec_results('P276_20140412_1734_15034019') 
#     plot_compare_spec_results('P276_20140412_1610_03669725')
#     plot_compare_spec_results('SPEC276_Full_Scale_20140414_1835_01089985')
    save_AE_output(model=None, preproc='GCN')
