
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

from smartNN.mlp import MLP
from smartNN.layer import RELU, Sigmoid, Softmax, Linear
from smartNN.datasets.mnist import Mnist
from smartNN.learning_rule import LearningRule
from smartNN.log import Log
from smartNN.train_object import TrainObject
from smartNN.cost import Cost

from smartNN.datasets.spec import P276_Spec

import os

import theano
import theano.tensor as T
import numpy as np

NNdir = os.path.dirname(os.path.realpath(__file__))

if not os.getenv('smartNN_DATA_PATH'):
    os.environ['smartNN_DATA_PATH'] = NNdir + '/data'

if not os.getenv('smartNN_SAVE_PATH'):
    os.environ['smartNN_SAVE_PATH'] = NNdir + '/save'

os.environ['PYTHONPATH'] += NNdir + '/smartNN'

print('smartNN_DATA_PATH = ' + os.environ['smartNN_DATA_PATH'])
print('smartNN_SAVE_PATH = ' + os.environ['smartNN_SAVE_PATH'])





def test():

#     h1_layer = Sigmoid(dim=100, W=None, b=None)
#     
#     output_layer = RELU(dim=10, W=None, b=None)
#  
#     mnist = Mnist(preprocess = None, 
#                     binarize = False,
#                     batch_size = 100,
#                     num_batches = None, 
#                     train_ratio = 5, 
#                     valid_ratio = 1,
#                     iter_class = 'SequentialSubsetIterator')
#     
#     mlp = MLP(layers = [output_layer], input_dim = mnist.feature_size())
#     
#     x = theano.tensor.matrix('x')
#     y = mlp.train_fprop(x)
#     
#     f = theano.function([x], y)
#     
#     print(mnist.get_train().X[0:2,300:400])
#     print(f(mnist.get_train().X[0:2]))
#     print(f(np.random.rand(1, 784)))

    W_update = T.matrix('x')
#     W_update = theano.shared(np.random.rand(100,10))
    
    
    w_len = T.sqrt((W_update ** 2).sum(axis=1))
    divisor = (w_len <= 2) + \
            (w_len > 2) * w_len
    update = W_update / divisor.reshape((divisor.shape[0], 1))
    
    f = theano.function([W_update], outputs=update)
#     print((f(np.random.rand(100,10))).shape)
#     print(W_update.get_value())
    

def spec():
    os.environ['smartNN_DATA_PATH'] = '/Applications/VCTK/data'
    
    data = P276_Spec(preprocess=None, feature_size=2049,
                batch_size=100, num_batches=None, 
                train_ratio=5, test_ratio=1,
                iter_class='ShuffledSequentialSubsetIterator')
                
    
    mlp = MLP(input_dim = data.feature_size())
    mlp.add_layer(RELU(dim=100, name='h1_layer', W=None, b=None))
    mlp.add_layer(Linear(dim=data.target_size(), name='output_layer', W=None, b=None))
    
    learning_rule = LearningRule(max_norm = 1,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                weight_decay = 0,
                                cost = Cost(type='mse'),
                                dropout = 1,
                                stopping_criteria = {'max_epoch' : 100, 
                                                    'epoch_look_back' : 3, 
                                                    'accu_increase' : 0.05}
                                )
    
    log = Log(experiment_id = 'testing',
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

    

def spec_autoencoder():
    
    learning_rule = LearningRule(max_norm = 0.1,
                            learning_rate = 0.1,
                            momentum = 0.01,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='mse'),
                            dropout = 0,
                            stopping_criteria = {'max_epoch' : 10, 
                                                'epoch_look_back' : None, 
                                                'accu_increase' : None}
                            )
    
    mnist = Mnist(preprocess = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
    
    train = mnist.get_train()
    mnist.set_train(train.X, train.X)
    
    mnist.valid = None
    mnist.test = None
    
#     valid = mnist.get_valid()
#     mnist.set_valid(valid.X, valid.X)
#     
#     test = mnist.get_test()
#     mnist.set_test(test.X, test.X)
    
    mlp = MLP(input_dim = mnist.feature_size(), rand_seed=None)
    h1_layer = RELU(dim=60, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    mlp.add_layer(Sigmoid(dim=28*28, name='output_layer', W=h1_layer.W.T, b=None))

    log = Log(experiment_id = 'testing',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = mnist,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()

def mlp():
     
    mnist = Mnist(preprocess = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
    
    mlp = MLP(input_dim = mnist.feature_size())
    mlp.add_layer(RELU(dim=100, name='h1_layer', W=None, b=None))
    mlp.add_layer(Linear(dim=10, name='output_layer', W=None, b=None))
    
    learning_rule = LearningRule(max_norm = 1,
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                weight_decay = 0,
                                cost = Cost(type='mse'),
                                dropout = 1,
                                stopping_criteria = {'max_epoch' : 100, 
                                                    'epoch_look_back' : 3, 
                                                    'accu_increase' : 0.05}
                                )
    
    log = Log(experiment_id = 'testing',
            description = 'This experiment is to test the model',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = mnist,
                                learning_rule = learning_rule,
                                log = log)
    train_object.run()
    
def autoencoder():
    
    learning_rule = LearningRule(max_norm = 2,
                            learning_rate = 0.1,
                            momentum = 0.01,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='mse'),
                            dropout = 0,
                            stopping_criteria = {'max_epoch' : 20, 
                                                'epoch_look_back' : None, 
                                                'accu_increase' : None}
                            )
    
    mnist = Mnist(preprocess = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
    
    train = mnist.get_train()
    mnist.set_train(train.X, train.X)
    
    mnist.valid = None
    mnist.test = None
    
#     valid = mnist.get_valid()
#     mnist.set_valid(valid.X, valid.X)
#     
#     test = mnist.get_test()
#     mnist.set_test(test.X, test.X)
    
    mlp = MLP(input_dim = mnist.feature_size(), rand_seed=None)
    h1_layer = RELU(dim=60, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    mlp.add_layer(Sigmoid(dim=28*28, name='output_layer', W=h1_layer.W.T, b=None))

    log = Log(experiment_id = 'testing',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    train_object = TrainObject(model = mlp,
                                dataset = mnist,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()
    
def stacked_autoencoder():

    #=====[ Train First layer of stack autoencoder ]=====#
    print('Start training First Layer of AutoEncoder')

    
    log = Log(experiment_id = 'testing',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')
    
    learning_rule = LearningRule(max_norm = 1,
                            learning_rate = 0.1,
                            momentum = 0.01,
                            momentum_type = 'normal',
                            weight_decay = 0,
                            cost = Cost(type='mse'),
                            dropout = 0,
                            stopping_criteria = {'max_epoch' : 10, 
                                                'epoch_look_back' : None, 
                                                'accu_increase' : None}
                            )

    mnist = Mnist(preprocess = None, 
                    binarize = False,
                    batch_size = 100,
                    num_batches = None, 
                    train_ratio = 5, 
                    valid_ratio = 1,
                    iter_class = 'SequentialSubsetIterator')
                    
    train = mnist.get_train()
    mnist.set_train(train.X, train.X)
    
#     valid = mnist.get_valid()
#     mnist.set_valid(valid.X, valid.X)
#     
#     test = mnist.get_test()
#     mnist.set_test(test.X, test.X)

    mnist.valid = None
    mnist.test = None
    
    mlp = MLP(input_dim = mnist.feature_size(), rand_seed=None)

    h1_layer = Sigmoid(dim=100, name='h1_layer', W=None, b=None)
    mlp.add_layer(h1_layer)
    h1_mirror = Sigmoid(dim=28*28, name='h1_mirror', W=h1_layer.W.T, b=None)
    mlp.add_layer(h1_mirror)

    
    train_object = TrainObject(model = mlp,
                                dataset = mnist,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()
    
    #=====[ Train Second Layer of autoencoder ]=====#
    
    print('Start training Second Layer of AutoEncoder')
    
    x = T.matrix('x')
    mlp.pop_layer(-1)
    reduced_X = mlp.fprop(x)
    f = theano.function([x], reduced_X)
    reduced_X = f(train.X)
    
    mnist.set_train(reduced_X, reduced_X)
    mnist.valid = None
    mnist.test = None
    
    mlp2 = MLP(input_dim = mnist.feature_size(), rand_seed=None)

    
    h2_layer = Sigmoid(dim=10, name='h2_layer', W=None, b=None)
    mlp2.add_layer(h2_layer)
    h2_mirror = Sigmoid(dim=h1_layer.dim, name='h2_mirror', W=h2_layer.W.T, b=None)
    mlp2.add_layer(h2_mirror)
    
              
    train_object = TrainObject(model = mlp2,
                            dataset = mnist,
                            learning_rule = learning_rule,
                            log = log)
    
    train_object.run()
    
    #=====[ Fine Tuning ]=====#
    
    print('Fine Tuning')
    
    mnist = Mnist(preprocess = None, 
                binarize = False,
                batch_size = 100,
                num_batches = None, 
                train_ratio = 5, 
                valid_ratio = 1,
                iter_class = 'SequentialSubsetIterator')
    
    train = mnist.get_train()
    
    mnist.set_train(train.X, train.X)
    mnist.valid = None
    mnist.test = None
    
    mlp3 = MLP(input_dim = mnist.feature_size(), rand_seed=None)
    mlp3.add_layer(h1_layer)
    mlp3.add_layer(h2_layer)
    mlp3.add_layer(h2_mirror)
    mlp3.add_layer(h1_mirror)
    
    
    train_object = TrainObject(model = mlp3,
                            dataset = mnist,
                            learning_rule = learning_rule,
                            log = log)
    
    train_object.run()

def savenpy():
    import glob
    os.environ['smartNN_DATA_PATH'] = '/Applications/VCTK/data'
    im_dir = os.environ['smartNN_DATA_PATH'] + '/inter-module/mcep/England/p276'
    
    files = glob.glob(im_dir + '/*.spec')
    
    size = 0
    data = np.asarray([], dtype='<f4')
    count = 0
    for f in files:
        if count > 100:
            break
        with open(f) as fb:
            clip = np.fromfile(fb, dtype='<f4', count=-1)
        data = np.concatenate([data, clip])
        print('..done ' + f)
        
        count += 1
        
        
        
    print(os.path.exists(im_dir))
    with open(im_dir + '/p276.npy', 'wb') as f:
        np.save(f, data)
    


if __name__ == '__main__':
    autoencoder()
#     mlp()
#     stacked_autoencoder()
#     spec()
#     savenpy()

                                
                                
                                
                         