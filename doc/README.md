
# Procedures for Reconstructing Spec Files with AutoEncoder #

In order to use this package, user should install Anaconda(a super package that includes 
numpy, matplotlib and others), Theano and sqlite3. And add smartNN directory to your PYTHONPATH.

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

After merging, two files are created, they are `p276_data_000.npy` which is a 2D data tensor of
dimension (num of frames, 2049) which is used for training and `p276_specnames_000.npy` 
which is a list of tuples of specification (name of specfile, num of frames in the specfile).

__2. Setting Environment Variables__

In smartNN, there are three environment variables to be set.

```python
smartNN_DATA_PATH   # the directory for all the datasets
smartNN_SAVE_PATH   # the directory to save the best models, the outputs logs and the hyperparameters 
smartNN_DATABASE_PATH # after training, the hyperparameters and training results from various 
                      # experiments is saved into a database for comparisions
``` 

__3. Building the Model__

In order to build and run an AutoEncoder, we need to put together the various components
(model, layer, dataset, learning_rule, log, cost function) into a train_object and run the
training. For more examples goto [AE_example.py](../example/AE_example.py)

```python
import theano
import theano.tensor as T
import numpy as np

from smartNN.model AutoEncoder
from smartNN.layer import RELU, Sigmoid, Softmax, Linear 
from smartNN.datasets.spec import P276
from smartNN.learning_rule import LearningRule
from smartNN.log import Log
from smartNN.train_object import TrainObject
from smartNN.cost import Cost
from smartNN.datasets.preprocessor import Standardize, GCN

def autoencoder():

    # logging is optional
    log = Log(experiment_name = 'AE',
            description = 'This experiment is about autoencoder',
            save_outputs = True,
            save_hyperparams = True,
            save_model = True,
            send_to_database = 'Database_Name.db')

    learning_rule = LearningRule(max_col_norm = None,
                            learning_rate = 0.01,
                            momentum = 0.1,
                            momentum_type = 'normal',
                            L1_lambda = None,
                            L2_lambda = None,
                            cost = Cost(type='mse'),
                            stopping_criteria = {'max_epoch' : 100,
                                                'cost' : Cost(type='mse'),
                                                'epoch_look_back' : 10,
                                                'percent_decrease' : 0.001}
                            )
    
    # building dataset, batch_size and preprocessor
    data = P276(train_valid_test_ratio=[5,1,1], batch_size=100, preprocessor=GCN())
    
    # for AutoEncoder, the inputs and outputs must be the same
    train = data.get_train()
    data.set_train(train.X, train.X)
    
    valid = data.get_valid()
    data.set_valid(valid.X, valid.X)
    
    test = data.get_test()
    data.set_test(test.X, test.X)
    
    # building autoencoder
    ae = AutoEncoder(input_dim = data.feature_size(), rand_seed=None)
    h1_layer = RELU(dim=100, name='h1_layer', W=None, b=None)
    
    # adding encoding layer
    ae.add_encode_layer(h1_layer)
    
    # adding decoding mirror layer, lock the weights of the output layer to be transpose of input layer
    ae.add_decode_layer(Sigmoid(dim=data.target_size(), name='output_layer', W=h1_layer.W.T, b=None))

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
`generate_specs_from_model.py` did the following things in one go 
1.  load model.pkl  
2.  load p276_data_000.npy
3.  preprocess p276_data_000.npy
4.  pass the preprocessed p276_data_000.npy through the model
5.  invert the output from the model (this is necessary because the output is from a preprocessed input)
6.  load p276_specnames_000.npy
7.  base on the specnames from p276_specnames_000.npy, reconstruct the specfiles from npy file

```python
import cPickle
import os
import numpy as np
from smartNN.datasets.spec import P276
from smartNN.datasets.preprocessor import GCN

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



