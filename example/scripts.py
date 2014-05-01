
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

from smartNN.model import MLP
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

        
    
def plot_compare_spec_results(model, proc): 
    import cPickle
    from smartNN.utils.utils import tile_raster_graphs
    from PIL.Image import fromarray
    import smartNN.datasets.preprocessor as processor
    
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

    start = -12
    end = -10
    tile_shape=(2,1)
    
    print('..loading data from ' + model)
    data = P276(train_valid_test_ratio=[8,1,1])
    
    test = data.get_test()
#     prep = Standardize()
#     prep = Scale()
    prep = getattr(processor, proc)()
    
    if prep.__class__.__name__ == 'Scale':
        prep.max = 98803.031
        prep.min = 0.1
    
    print('..preprocessing data ' + prep.__class__.__name__ )
    proc_test_X = prep.apply(test.X)
    print('..fprop X')
    output = mlp.fprop(proc_test_X)
    print('..saving data')
    plt = tile_raster_graphs(proc_test_X[start:end], output[start:end], slice=(0,-1),
                            tile_shape=tile_shape, tile_spacing=(0.1,0.1), legend=True)
    
    plt.savefig(os.environ['smartNN_SAVE_PATH'] + '/images/' + model + '_0_2049.png')
    
    plt.close()
    
    plt = tile_raster_graphs(proc_test_X[-12:-10], output[-12:-10], slice=(0,200),
                            tile_shape=(2,1), tile_spacing=(0.1,0.1), legend=True)
    
    plt.savefig(os.environ['smartNN_SAVE_PATH'] + '/images/' + model + '_0_200.png')
    
    plt.close()
    plt = tile_raster_graphs(proc_test_X[-12:-10], output[-12:-10], slice=(1500,-1),
                            tile_shape=(2,1), tile_spacing=(0.1,0.1), legend=True)
    
    plt.savefig(os.environ['smartNN_SAVE_PATH'] + '/images/' + model + '_1500_2400.png')
    plt.close()
    print('Saved Successfully')
#     plt.show()


def extract_examples(folder_path, start, end):
    files = glob.glob(folder_path + '/*.spec')

    files = files[start:end]
    data = []
    count = 0
    size = len(files)
    for f in files:
        with open(f) as fb:
            clip = np.fromfile(fb, dtype='<f8', count=-1)
            data.extend(clip)

        print(str(count) + '/' + str(size) + '..done '  + f)
        
        count += 1
    data = np.asarray(data)
    data = data.reshape(data.shape[0]/2049, 2049)

    return data

def plot_compare_spec_results2(model, proc): 
    import cPickle
    from smartNN.utils.utils import tile_raster_graphs
    from PIL.Image import fromarray
    import smartNN.datasets.preprocessor as processor
    
    with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + model + '/model.pkl', 'rb') as f:
        mlp = cPickle.load(f)
    
    
    start = -12
    end = -10
    tile_shape=(2,1)
    
    print('..loading data from spec.double')
    folder_path = os.environ['smartNN_DATA_PATH'] + '/p276_double'
    dct_data = extract_examples(folder_path, -10, -1)
    dct_data = dct_data[start:end]
    
    
    
    print('..loading data from ' + model)
    data = P276(train_valid_test_ratio=[8,1,1])
    
    test = data.get_test()
#     prep = Standardize()
#     prep = Scale()
    prep = getattr(processor, proc)()
    
    if prep.__class__.__name__ == 'Scale':
        prep.max = 98803.031
        prep.min = 0.1
    
    print('..preprocessing data ' + prep.__class__.__name__ )
    proc_test_X = prep.apply(test.X)
    dct_data = prep.apply(dct_data)
    print('..fprop X')
    output = mlp.fprop(proc_test_X)
    print('..saving data')

    plt = tile_raster_graphs(dct_data, proc_test_X[start:end], output[start:end], slice=(0,-1),
                            tile_shape=tile_shape, tile_spacing=(0.1,0.1), legend=True)
    
    plt.savefig(os.environ['smartNN_SAVE_PATH'] + '/images/' + model + '_0_2049_all2.png')
    
    plt.close()
    
    plt = tile_raster_graphs(dct_data, proc_test_X[-12:-10], output[-12:-10], slice=(0,200),
                            tile_shape=(2,1), tile_spacing=(0.1,0.1), legend=True)
    
    plt.savefig(os.environ['smartNN_SAVE_PATH'] + '/images/' + model + '_0_200_all2.png')
    
    plt.close()
    plt = tile_raster_graphs(dct_data, proc_test_X[-12:-10], output[-12:-10], slice=(1500,-1),
                            tile_shape=(2,1), tile_spacing=(0.1,0.1), legend=True)
    
    plt.savefig(os.environ['smartNN_SAVE_PATH'] + '/images/' + model + '_1500_2400_all2.png')
    plt.close()
    print('Saved Successfully')
#     plt.show()

def save_AE_output(model, preproc):
    import cPickle
    from PIL.Image import fromarray
    import smartNN.datasets.preprocessor as proc
    
    with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + model + '/model.pkl', 'rb') as f:
        mlp = cPickle.load(f)
    
    prep = getattr(proc, preproc)()
    
    print('..loading data from ' + model)
    data = P276(train_valid_test_ratio=[1,0,0])
#     data = Mnist(train_valid_test_ratio=[1,0,0])
    train = data.get_train()
    print('..applying preprocessing')
    train.X = prep.apply(train.X)
    print('..fprop')
    mlp.pop_layer(-1)

    out = mlp.fprop(train.X)
    print('..save AE outputs')
    with open(os.environ['smartNN_DATA_PATH'] + '/p276/' + 'p276_' 
            + preproc + '_AE_output.npy', 'wb') as f:
        np.save(f, out)
    print('..output shape ' + str(out.shape))
    print('completed successfully')



    
    
    

if __name__ == '__main__':

#     savenpy('/RQusagers/hycis/smartNN/data/Laura')
#     testpreproc('/Applications/VCTK/data/inter-module/mcep/England/p276/p276.npy')
#     testpreproc('/data/lisa/exp/wuzhen/smartNN/data/p276/p276.npy')
#     processnpy('/Applications/VCTK/data/inter-module/mcep/England/p276/p276.npy') 
#     plot_compare_spec_results('P276_20140412_1734_15034019') 
#     plot_compare_spec_results('P276_20140412_1610_03669725')
#     plot_compare_spec_results('SPEC276_Full_Scale_20140414_1835_01089985')
#     plot_compare_spec_results('AE15_GCN_20140414_2342_39424209', 'GCN')
#     plot_compare_spec_results('AE15Double_GCN_20140415_1404_50336696', 'GCN')
    plot_compare_spec_results2('AE15Double_GCN_20140415_1807_28490384', 'GCN')
#     plot_compare_spec_results2('AE15Double_Scale_20140415_1209_15009927', 'Scale')
#     plot_compare_spec_results('AE15_Scale_20140414_2350_55857133', 'Scale')
#     plot_compare_spec_results('AE15_Scale_20140414_2349_19835883', 'Scale')
#     save_AE_output(model='AE15_GCN_20140414_2342_39424209', preproc='GCN')
#     save_AE_output(model='AE15_Scale_20140414_2349_19835883', preproc='Scale')
#     extract_examples('/RQusagers/hycis/smartNN/data/p276_double', -12, -10)
