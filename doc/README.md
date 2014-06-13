
# Procedures for Reconstructing Spec Files with AutoEncoder #

In order to use this package, user should install Anaconda(a super package that includes 
numpy, matplotlib and others), Theano and sqlite3. And add smartNN directory to your PYTHONPATH.

Steps from data preparation to training model to generating specs from model

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

__2. Setting Environment Variables__

In smartNN, there are three environment variables to be set.

```python
smartNN_DATA_PATH   # the directory for all the datasets
smartNN_SAVE_PATH   # the directory to save the best models, the outputs logs and the hyperparams 
smartNN_DATABASE_PATH # the directory to save the database which contains the stats from 
                      # all the experiments which is used for picking the best model
``` 

__3. Building the Model__

To build AutoEncoder, first import the 

```python
import theano
import theano.tensor as T
import numpy as np

from smartNN.model AutoEncoder # import AutoEncoder model
from smartNN.layer import RELU, Sigmoid, Softmax, Linear 
from smartNN.datasets.mnist import Mnist
from smartNN.learning_rule import LearningRule
from smartNN.log import Log # OPTIONAL, if you don't want logging, then this not necessary.
from smartNN.train_object import TrainObject
from smartNN.cost import Cost
from smartNN.datasets.preprocessor import Standardize, GCN

def autoencoder():
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
    
    # building dataset
    data = Mnist(train_valid_test_ratio=[5,1,1])
    
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

    train_object = TrainObject(model = ae,
                                dataset = data,
                                learning_rule = learning_rule,
                                log = log)
                                
    train_object.run()

```




