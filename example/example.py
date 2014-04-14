
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
from smartNN.datasets.preprocessor import Standardize, GCN

from smartNN.datasets.spec import P276


def mlp(state, channel):
     
#     data = Mnist(preprocessor = Standardize(can_fit=True), 
#                     binarize = False,
#                     batch_size = 100,
#                     num_batches = None, 
#                     train_ratio = 5, 
#                     valid_ratio = 1,
#                     iter_class = 'SequentialSubsetIterator')

    data = P276(feature_size=2049, train_ratio=8, 
                    valid_ratio=1, test_ratio=1)
    
#     data = Mnist(preprocessor = None, 
#                     binarize = False,
#                     batch_size = 100,
#                     num_batches = None, 
#                     train_ratio = 5, 
#                     valid_ratio = 1,
#                     iter_class = 'SequentialSubsetIterator')
    
    mlp = MLP(input_dim = data.feature_size())
    mlp.add_layer(RELU(dim=10, name='h1_layer', W=None, b=None))
    mlp.add_layer(RELU(dim= data.target_size(), name='output_layer', W=None, b=None))
    
    learning_rule = LearningRule(max_col_norm = 0.1,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                weight_decay = 0,
                                cost = Cost(type='mse'),
                                stopping_criteria = {'max_epoch' : 5, 
                                                    'epoch_look_back' : 3,
                                                    'cost' : Cost(type='mse'), 
                                                    'percent_decrease' : 0.001}
                                )
    
    log = Log(experiment_id = 'lahlah2',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
    train_object.run()
    return channel.COMPLETE

    
def autoencoder():
    
    learning_rule = LearningRule(max_col_norm = None,
                            learning_rate = 0.01,
                            momentum = 0.1,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='entropy'),
                            stopping_criteria = {'max_epoch' : 10,
                                                'cost' : Cost(type='entropy'),
                                                'epoch_look_back' : 3,
                                                'percent_decrease' : 0.001}
                            )
    
    data = Mnist(preprocessor = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)
    
    mlp = MLP(input_dim = data.feature_size(), rand_seed=None)
    h1_layer = RELU(dim=100, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    mlp.add_layer(Sigmoid(dim=data.target_size(), name='output_layer', W=h1_layer.W.T, b=None))

    log = Log(experiment_id = 'AE',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()
    
def stacked_autoencoder():

    name = 'stacked_AE4'

    #=====[ Train First layer of stack autoencoder ]=====#
    print('Start training First Layer of AutoEncoder')

    
    log = Log(experiment_id = name + '_layer1',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    learning_rule = LearningRule(max_col_norm = None,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                weight_decay = 0,
                                cost = Cost(type='entropy'),
                                stopping_criteria = {'max_epoch' : 1000, 
                                                    'epoch_look_back' : 10,
                                                    'cost' : Cost(type='entropy'), 
                                                    'percent_decrease' : 0.001}
                                )

    data = Mnist(preprocessor = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator',
                    rng = None)
                    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)

#     data.valid = None
#     data.test = None
    
    mlp = MLP(input_dim = data.feature_size(), rand_seed=None)

    h1_layer = Sigmoid(dim=500, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    h1_mirror = Sigmoid(dim = data.target_size(), name='h1_mirror', W=h1_layer.W.T, b=None)
    mlp.add_layer(h1_mirror)

    
    train_object = TrainObject(model = mlp,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()
    
    #=====[ Train Second Layer of autoencoder ]=====#
    
    log2 = Log(experiment_id = name + '_layer2',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    learning_rule = LearningRule(max_col_norm = None,
                            learning_rate = 0.01,
                            momentum = 0.1,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='entropy'),
                            stopping_criteria = {'max_epoch' : 1000, 
                                                'epoch_look_back' : 10,
                                                'cost' : Cost(type='entropy'), 
                                                'percent_decrease' : 0.001}
                            )

    
    print('Start training Second Layer of AutoEncoder')
    
    mlp.pop_layer(-1)
    reduced_train_X = np.abs(mlp.fprop(train.X))
    reduced_valid_X = np.abs(mlp.fprop(valid.X))
    reduced_test_X = np.abs(mlp.fprop(test.X))
#     import pdb
#     pdb.set_trace()
    data.set_train(reduced_train_X, reduced_train_X)
    data.set_valid(reduced_valid_X, reduced_valid_X)
    data.set_test(reduced_test_X, reduced_test_X)
    
    # create a new mlp taking inputs from the encoded outputs of first autoencoder
    mlp2 = MLP(input_dim = data.feature_size(), rand_seed=None)

    
    h2_layer = Sigmoid(dim=100, name='h2_layer', W=None, b=None)
    mlp2.add_layer(h2_layer)
    
    h2_mirror = Sigmoid(dim=h1_layer.dim, name='h2_mirror', W=h2_layer.W.T, b=None)
    mlp2.add_layer(h2_mirror)
    
              
    train_object = TrainObject(model = mlp2,
                            dataset = data,
                            learning_rule = learning_rule,
                            log = log2)
    
    train_object.run()
    
    #=====[ Fine Tuning ]=====#
    
    log3 = Log(experiment_id = name + '_full',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    print('Fine Tuning')
    
    data = Mnist(preprocessor = None, 
                binarize = False,
                batch_size = 100,
                num_batches = None, 
                train_ratio = 5, 
                valid_ratio = 1,
                iter_class = 'SequentialSubsetIterator')
    
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)
    
    mlp3 = MLP(input_dim = data.feature_size(), rand_seed=None)
    mlp3.add_layer(h1_layer)
    mlp3.add_layer(h2_layer)
    mlp3.add_layer(h2_mirror)
    mlp3.add_layer(h1_mirror)
    
    
    train_object = TrainObject(model = mlp3,
                            dataset = data,
                            learning_rule = learning_rule,
                            log = log3)
    
    train_object.run()
    print('..Training Done')

def savenpy(folder_path):
    import glob
    import itertools
    import matplotlib.pyplot as plt
    
    files = glob.glob(folder_path + '/*.spec')
    size = len(files)
    data = []
    count = 0
    for f in files:
        with open(f) as fb:
            print('..file: ', f)
            clip = np.fromfile(fb, dtype='<f4', count=-1)
            data.extend(clip)
            plt.plot(clip[50:2049])
            plt.show()
            import pdb
            pdb.set_trace()
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
#     autoencoder()
#     mlp()
#     stacked_autoencoder()
#     spec()
#     savenpy('/Applications/VCTK/data/inter-module/mcep/England/p276')
#     test()
    unpickle_mlp('stacked_AE5_20140410_0707_50423148')
#     test_AE()
                                
                                
                                
                         