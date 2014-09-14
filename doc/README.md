

__1. Setting Environment Variables__

In pynet, there are three environment variables to be set.

```python
PYNET_DATA_PATH   # the directory for all the datasets
PYNET_SAVE_PATH   # the directory to save the best models, the outputs logs and the hyperparameters 
PYNET_DATABASE_PATH # after training, the hyperparameters and training results from various 
                      # experiments is saved into a database for comparisions
``` 

__2. Building the Model__

In order to build and run an AutoEncoder, we need to put together the various components
(model, layer, dataset, learning_rule, log, cost function) into a train_object and run the
training. For more examples goto [example](../example/).

```python
import theano
import theano.tensor as T
import numpy as np

from pynet.model AutoEncoder
from pynet.layer import RELU, Sigmoid, Softmax, Linear 
from pynet.datasets.spec import *
from pynet.learning_rule import LearningRule
from pynet.log import Log
from pynet.train_object import TrainObject
from pynet.cost import Cost
from pynet.datasets.preprocessor import Standardize, GCN

def autoencoder():

    # set environment
    NNdir = os.path.dirname(os.path.realpath(__file__))
    NNdir = os.path.dirname(NNdir)
    NNdir = os.path.dirname(NNdir)

    if not os.getenv('PYNET_DATA_PATH'):
        os.environ['PYNET_DATA_PATH'] = NNdir + '/data'

    if not os.getenv('PYNET_DATABASE_PATH'):
        os.environ['PYNET_DATABASE_PATH'] = NNdir + '/database'
        if not os.path.exists(os.environ['PYNET_DATABASE_PATH']):
            os.mkdir(os.environ['PYNET_DATABASE_PATH'])

    if not os.getenv('PYNET_SAVE_PATH'):
        os.environ['PYNET_SAVE_PATH'] = NNdir + '/save'
        if not os.path.exists(os.environ['PYNET_SAVE_PATH']):
            os.mkdir(os.environ['PYNET_SAVE_PATH'])


    # logging is optional, it is used to save the best trained model and records the training result to a database
    log = Log(experiment_name = 'AE',
            description = 'This experiment is about autoencoder',
            save_outputs = True, # saves to outputs.log
            save_hyperparams = True,
            save_model = True,
            save_to_database = {'name': 'Example.db',
                                'records' : {'Dataset' : data.__class__.__name__,
                                             'Weight_Init_Seed' : mlp.rand_seed,
                                             'Dropout_Below' : str([layer.dropout_below for layer in mlp.layers]),
                                             'Batch_Size' : data.batch_size,
                                             'Layer_Size' : len(mlp.layers),
                                             'Layer_Dim' : str([layer.dim for layer in mlp.layers]),
                                             'Preprocessor' : data.preprocessor.__class__.__name__,
                                             'Learning_Rate' : learning_rule.learning_rate,
                                             'Momentum' : learning_rule.momentum}}
            ) # end log


    learning_rule = LearningRule(max_col_norm = None, # max length of the weight vector from lower layer going into upper neuron
                                learning_rate = 0.01,
                                momentum = 0.1,
                                momentum_type = 'normal',
                                L1_lambda = None, # L1 regularization coefficient
                                L2_lambda = None, # L2 regularization coefficient
                                cost = Cost(type='mse'), # cost type use for backprop during training
                                stopping_criteria = {'max_epoch' : 100, # maximum number of epochs for the training
                                                    'cost' : Cost(type='mse'), # cost type use for testing the quality of the trained model
                                                    'epoch_look_back' : 10, # number of epoch to look back for error improvement
                                                    'percent_decrease' : 0.001} # requires at least 0.001 = 0.1% decrease in error when look back of 10 epochs
                                )
                            
    
    # building dataset, batch_size and preprocessor
    data = Laura_Blocks(train_valid_test_ratio=[8,1,1], batch_size=100, preprocessor=GCN())
    
    # for AutoEncoder, the inputs and outputs must be the same
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)
    
    # building autoencoder
    ae = AutoEncoder(input_dim = data.feature_size(), rand_seed=None)
    h1_layer = Tanh(dim=500, name='h1_layer', W=None, b=None)
    
    # adding encoding layer
    ae.add_encode_layer(h1_layer)
    
    # mirror layer has W = h1_layer.W.T
    h1_mirror = Tanh(name='h1_mirror', W=h1_layer.W.T, b=None)
    
    # adding decoding mirror layer
    ae.add_decode_layer(h1_mirror)

    # put all the components into a TrainObject
    train_object = TrainObject(model = ae,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
    
    # finally run the training                         
    train_object.run()
    
```


