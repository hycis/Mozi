
Below is the basic structure for pretraining of one layer autoencoder,
there are basically following hyperparams that will affect the training result

1. max_col_norm
2. learning_rate
3. momentum
4. batch_size
5. rand_seed # the seed for initializing the weights in autoencoder
6. dropout_below # the dropout below for the denoising autoencoder

By setting the hyperparams and run the Model Script below, it can generates one result.


__1. Setting Environment Variables__

In pynet, there are three environment variables to be set.

```python
PYNET_DATA_PATH   # the directory for all the datasets
PYNET_SAVE_PATH   # the directory to save the best models, the outputs logs and the hyperparameters
PYNET_DATABASE_PATH # after training, the hyperparameters and training results from various
                      # experiments is saved into a database for comparisions
```

__2. Model Script__

In order to build and run an AutoEncoder, we need to put together the various components
(model, layer, dataset, learning_rule, log, cost function) into a train_object and run the
training. The example model below is saved to the script [AE_example.py](../example/AE_example.py).

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


    learning_rule = LearningRule(max_col_norm = 1, # max length of the weight vector from lower layer going into upper neuron
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
    ae = AutoEncoder(input_dim = data.feature_size(), rand_seed=123)
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

__3. Hyperparams Search__

In order to do hyperparams search, run the script in [launch.py](../hps/launch.py) in [hps dir](../hps).
To do that, first log into helios

```bash
ssh hycis@helios.calculquebec.ca
cdwu # change to the scratch directory
cd Pynet/hps
cat model_config.py # this will show the configurations of different models
```

Inside model_config.py, if the values is placed in a tuple for a variable,
it means that during the sampling of values for a variable,
the value are sampled uniformly from the values in the tuple.
For example for
```'learning_rate' : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5)```,
learning_rate is uniformly set as any of the 6 values in the tuple.

Below is the sample of model Laura from [model_config.py](../hps/model_config.py)  
```python
    'Laura' : DD({
            'model' : DD({
                    'rand_seed'             : None
                    }), # end mlp

            'log' : DD({
                    'experiment_name'       : 'AE0918_Warp_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                    # 'experiment_name'       : 'AE0914_Warp_Blocks_500_180_tanh_tanh_gpu_clean', #helios
            
                    # 'experiment_name'       : 'AE0919_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                    # 'experiment_name'       : 'AE0918_Blocks_180_120_tanh_tanh_gpu_clean', #helios

                    # 'experiment_name'       : 'AE0916_Blocks_180_120_tanh_tanh_gpu_output_sig_dropout',
                    # 'experiment_name'       : 'AE0916_Blocks_180_120_tanh_tanh_gpu_output_sig_clean',


                    'description'           : '',
                    'save_outputs'          : True,
                    'save_hyperparams'      : True,
                    'save_model'            : True,
                    'save_to_database_name' : 'Laura.db'
                    }), # end log


            'learning_rule' : DD({
                    'max_col_norm'          : (1, 10, 50),
                    'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                    'momentum'              : (1e-3, 1e-2, 1e-1, 0.5, 0.9),
                    'momentum_type'         : 'normal',
                    'L1_lambda'             : None,
                    'L2_lambda'             : None,
                    'cost'                  : 'mse',
                    'stopping_criteria'     : DD({
                                                'max_epoch'         : 100,
                                                'epoch_look_back'   : 10,
                                                'cost'              : 'mse',
                                                'percent_decrease'  : 0.05
                                                }) # end stopping_criteria
                    }), # end learning_rule

            #===========================[ Dataset ]===========================#
            'dataset' : DD({
                    # 'type'                  : 'Laura_Warp_Blocks_500_Tanh',
                    'type'                 : 'Laura_Warp_Blocks_180_Tanh_Dropout',
                    # 'type'                  : 'Laura_Cut_Warp_Blocks_300',
                    # 'type'                  : 'Laura_Blocks_180_Tanh_Tanh',
                    # 'type'                  : 'Laura_Blocks_180_Tanh_Tanh_Dropout',
                    # 'type'                  : 'Laura_Blocks_500_Tanh_Sigmoid',
                    # 'type'                  : 'Laura_Blocks_500',
                    # 'type'                  : 'Laura_Blocks',
                    # 'type'                  : 'Laura_Warp_Blocks',
                    # 'type'                  : 'Laura_Warp_Standardize_Blocks',
                    # 'type'                  : 'Laura_Standardize_Blocks',
                    # 'type'                  : 'Mnist',

                    'feature_size'          : 180,
                    'train_valid_test_ratio': [8, 1, 1],

                    'preprocessor'          : None,
                    # 'preprocessor'          : 'Scale',
                    # 'preprocessor'          : 'GCN',
                    # 'preprocessor'          : 'LogGCN',
                    # 'preprocessor'          : 'Standardize',

                    'batch_size'            : (50, 100, 150, 200),
                    'num_batches'           : None,
                    'iter_class'            : 'SequentialSubsetIterator',
                    'rng'                   : None
                    }), # end dataset

            #============================[ Layers ]===========================#
            'num_layers' : 1,

            'hidden1' : DD({
                    'name'                  : 'hidden1',
                    'type'                  : 'Tanh',
                    'dim'                   : 120,

                    # 'dropout_below'         : None,
                    'dropout_below'         : (0.1, 0.2, 0.3, 0.4, 0.5),
                    # 'dropout_below'         : 0.5,

                    }), # end hidden_layer

            'hidden2' : DD({
                    'name'                  : 'hidden2',
                    'type'                  : 'RELU',
                    'dim'                   : 100,
                    'dropout_below'         : None,
                    }), # end hidden_layer

            'h2_mirror' : DD({
                    'name'                  : 'h2_mirror',
                    'type'                  : 'RELU',
                    # 'dim'                   : 2049, # dim = input.dim
                    'dropout_below'         : None,
                    }), # end output_layer

            'h1_mirror' : DD({
                    'name'                  : 'h1_mirror',
                    'type'                  : 'Tanh',
                    # 'dim'                   : 2049, # dim = input.dim
                    'dropout_below'         : None,
                    }) # end output_layer

            }), # end autoencoder
```


To sample one set of hyperparams and run it locally, issue
```bash
cdwu
cd Pynet/hps
python launch.py --model Laura -c 1
```
To submit 5 jobs to the gpu cluster, issue
```bash
cdwu
cd Pynet/hps
python launch.py --model Laura -n 5 -g
showq -u hycis
```

After finished running, you can checkout the results from the database
```bash
cdwu
sqlite3 Pynet/database/Laura.db
>>> .header on
>>> .mode column
>>> .table
>>> select * from some_table order by test_error;
```

I have named the the experiment group in as way that is easier for understanding, for example
for an experiment group name of ```AE0912_Blocks_2049_500_tanh_tanh_gpu_clean```
means AE0912 trained on Linear Blocks of autoencoder with 2049-500-2049 dims, and tanh-tanh units,
it's run on gpu and it's a clean model without noise during training.
The best model for the experiment group is ```AE0912_Blocks_2049_500_tanh_tanh_gpu_clean_20140914_1242_27372903```
where the last few numbers are the actual date_time_microsec in which the model is generated.


I have saved the best results for each pretrain layer in the http://1drv.ms/1qSyrZI under the *combinations* section.

__4. Reproduce Best Results__

To reproduce the results you can plug the hyperparams saved in the database into [AE_example.py](../example/AE_example.py)
and run the job locally, or you can modify the [model_config.py](../hps/model_config.py) and
set the hyperparams in the config file and run ```python launch.py --model Laura -c 1```

*Stacking up Models*

To reproduce the stackup of trained model is very simple. Just put the name of the best 
model under 'hidden1' and 'hidden2' in the model_config.py and set the hyperparams, and issue 
```python launch.py --model Laura_Two_Layers -c 1``` to run the job locally. 

```python
'Laura_Two_Layers' : DD({
        'model' : DD({
                'rand_seed'             : None
                }), # end mlp

        'log' : DD({
                # 'experiment_name'       : 'AE0917_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_clean',
                # 'experiment_name'       : 'AE0918_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy',

                # 'experiment_name'       : 'AE0918_Blocks_2layers_finetune_2049_180_tanh_sigmoid_gpu_clean',
                # 'experiment_name'       : 'AE0917_Blocks_2layers_finetune_2049_180_tanh_sigmoid_gpu_noisy',

                # 'experiment_name'       : 'AE0917_Warp_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_clean',
                'experiment_name'       : 'AE0918_Warp_Blocks_2layers_finetune_2049_180_tanh_tanh_gpu_noisy',



                'description'           : '',
                'save_outputs'          : True,
                'save_hyperparams'      : True,
                'save_model'            : True,
                'save_to_database_name' : 'Laura.db'
                }), # end log


        'learning_rule' : DD({
                'max_col_norm'          : (1, 10, 50),
                'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                # 'learning_rate'         : ((1e-5, 9e-1), float),
                # 'learning_rate'         : 0.01,
                'momentum'              : (1e-3, 1e-2, 1e-1, 0.5, 0.9),
                # 'momentum'              : 0.05,
                'momentum_type'         : 'normal',
                'L1_lambda'             : None,
                'L2_lambda'             : None,
                'cost'                  : 'mse',
                'stopping_criteria'     : DD({
                                            'max_epoch'         : 100,
                                            'epoch_look_back'   : 10,
                                            'cost'              : 'mse',
                                            'percent_decrease'  : 0.05
                                            }) # end stopping_criteria
                }), # end learning_rule

        #===========================[ Dataset ]===========================#
        'dataset' : DD({
                # 'type'                  : 'Laura_Warp_Blocks_500',
                # 'type'                  : 'Laura_Blocks_500',
                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',
                # 'type'                  : 'Mnist_Blocks',
                'feature_size'          : 2049,
                'train_valid_test_ratio': [8, 1, 1],

                # 'preprocessor'          : None,
                # 'preprocessor'          : 'Scale',
                'preprocessor'          : 'GCN',
                # 'preprocessor'          : 'LogGCN',
                # 'preprocessor'          : 'Standardize',

                'batch_size'            : (50, 100, 150, 200),
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        # #============================[ Layers ]===========================#

        'hidden1' : DD({
                'name'                  : 'hidden1',

                # 'model'                 : 'AE0912_Blocks_2049_500_tanh_tanh_gpu_clean_20140914_1242_27372903',
                # 'model'                 : 'AE0915_Blocks_2049_500_tanh_tanh_gpu_Dropout_20140915_1900_37160748',

                # 'model'                 : 'AE0912_Blocks_2049_500_tanh_sigmoid_gpu_clean_20140913_1342_18300926',

                # 'model'                 : 'AE0911_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140912_2337_04263067',
                'model'                 : 'AE0916_Warp_Blocks_2049_500_tanh_tanh_gpu_dropout_20140916_1705_29139505',


                'dropout_below'         : None,
                # 'dropout_below'         : 0.1,
                }), # end hidden_layer

        'hidden2' : DD({
                'name'                  : 'hidden2',
                # 'model'                 : 'AE0916_Blocks_500_180_tanh_tanh_gpu_clean_20140916_2255_06553688',
                # 'model'                 : 'AE0914_Blocks_500_180_tanh_tanh_gpu_dropout_20140916_1059_59760060',
                # 'model'                 : 'AE0918_Blocks_500_180_tanh_tanh_gpu_dropout_20140918_0920_42738052',

                # 'model'                 : 'AE0916_Blocks_500_180_tanh_tanh_gpu_output_sig_clean_20140917_0301_44075773',

                # 'model'                 : 'AE0914_Warp_Blocks_500_180_tanh_tanh_gpu_clean_20140915_0400_30113212',
                # 'model'                 : 'AE0916_Warp_Blocks_500_180_tanh_tanh_gpu_dropout_20140916_1326_09742695',
                'model'                 : 'AE0918_Warp_Blocks_500_180_tanh_tanh_gpu_dropout_20140918_1125_23612485',

                'dropout_below'         : None,
                }), # end hidden_layer


        }), # end autoencoder
```



