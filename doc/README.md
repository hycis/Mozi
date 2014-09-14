
# Procedures for Reconstructing Spec Files with AutoEncoder #

In order to use this package, user should install Anaconda(a super package that includes 
numpy, matplotlib and others), Theano and sqlite3. And add pynet directory to your PYTHONPATH.

Steps from data preparation to model training to generating specs from model

__1. Generate datafile from spec files__

In order to feed the data into the training framework, 
the first step is to merge all the spec files into a numpy data 
file that is readable by AutoEncoder by using the script
[specs2data.py](../scripts/specs2data.py)

In order to know all the options available for the script, use 

```bash
$ python specs2data.py -h
```

For example, in order to merge p276 spec files into one npy file (splits = 1), issue

```bash
$ python specs2data.py --spec_files /path/to/p276/*.spec --splits 1 --input_spec_dtype f4 
--feature_size 2049 --output_dir /path/to/output_dir/
```

After merging, two files are created, they are `p276_data_000.npy` which is a 2D data tensor 
of dimension (num of frames, 2049) and `p276_specnames_000.npy` 
which is a list of tuples of specification (name of specfile, num of frames in the specfile).

The last three number in `p276_data_000.npy` corresponds to the split id of that data file.
`p276_specnames_000` is used for unrolling the specfiles from the npy data file after training.

__2. Setting Environment Variables__

In pynet, there are three environment variables to be set.

```python
PYNET_DATA_PATH   # the directory for all the datasets
PYNET_SAVE_PATH   # the directory to save the best models, the outputs logs and the hyperparameters 
PYNET_DATABASE_PATH # after training, the hyperparameters and training results from various 
                      # experiments is saved into a database for comparisions
``` 

__3. Building the Model__

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


    # logging is optional
    log = Log(experiment_name = 'AE',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')

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
__4. Generate Outputs from the saved Model__

After training, pick the best model which is saved as a pickle file. And run test data through it
to generate the results. To generate all the specfiles from the model. Run the script
[generate_specs_from_model.py](../scripts/generate_specs_from_model.py).

```bash
$ python generate_specs_from_model.py --model /path/to/model.pkl --preprocessor GCN 
--dataset /path/to/p276_data_000.npy --output_dir /dir/for/specfiles/ --output_dtype <f8
```
The script `generate_specs_from_model.py` basically goes through the following steps    
1.  load model.pkl  
2.  load p276_data_000.npy  
3.  preprocess p276_data_000.npy  
4.  pass the preprocessed p276_data_000.npy through the model  
5.  invert the output from the model (this is necessary because the output is from a preprocessed input)  
6.  load p276_specnames_000.npy  
7.  base on the specnames from p276_specnames_000.npy, reconstruct the specfiles from npy file  

Below shows the steps from 1 to 5. Step 6 and 7 can also be done separately using 
[data2specs.py](../scrips/data2specs.py)

```python
import cPickle
import os
import numpy as np
from pynet.datasets.spec import P276
from pynet.datasets.preprocessor import GCN

# If there is preprocessing before training, then before passing the test data through the model,
# it has to be preprocessed also.
with open('/path/to/p276_data_000.npy') as d:
  dataset_raw = np.load(d)
proc = GCN()
print 'apply preprocessing..'
dataset_proc = proc.apply(dataset_raw)
del dataset_raw

# unpickle the trained model
print 'opening model.. '
with open('/path/to/model.pkl') as m:
  model = cPickle.load(m)

# pass the processed data through the model
print 'forward propagation..'
dataset_out = model.fprop(dataset_proc)
del dataset_proc

# invert the preprocessing
print 'invert dataset..'
dataset = proc.invert(dataset_out)
dataset = dataset.astype('<f8')
del dataset_out

np.save(dataset, '/path/to/p276_data_000_proc.npy')
```



