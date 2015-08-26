Mozi
=====

Mozi is based on Theano with a clean and sharp design, the **design philosophy** of Mozi

1. **Fast and Simple**: The main engine of the package is only 200 lines of code. There is only one full compiled graph for training which ensures all the data manipulation happens in one go through the pipeline and as a result make it super fast package.
2. **Highly Modular**: Building a model in Mozi is like building a house with Lego, you can design whatever imaginable layers and stack them together easily.
3. **Model Abstract from Training**: In order to facilitate deployment of trained model for real use, the model is abstracted away from the training module and keep as minimalist as possible. Objective is to allowed realtime deployment and easy model exchange.
4. **Logging System**: Mozi provides a full logging feature that allows user to log the training results and the hyperparameters to the database for paranormal overview. Also it allows automatic saving of best model and logging of all training outputs for easy aftermath analysis.

---
### Set Environment
In Mozi, we need to set three environment paths
* *MOZI_DATA_PATH*
* *MOZI_SAVE_PATH*
* *MOZI_DATABASE_PATH*

`MOZI_DATA_PATH` is the directory for saving and loading the datasets.  
`MOZI_SAVE_PATH` is the directory for saving all the models, the output log and epoch error.  
`MOZI_DATABASE_PATH` is the directory for saving the database that contains tables recording the hyperparameters and test errors for each training job.

---
### Let's Have Fun!
Building a model in Mozi is as simple as

```python
import theano.tensor as T
from mozi.model import Sequential
model = Sequential(input_var=T.matrix(), output_var=T.matrix())
```
The `input_var` is the input tensor variable that corresponds to the number of dimensions of the input dataset. The `output_var` is the tensor variable that corresponds to the target of the dataset. `T.matrix` for `2d data`, `T.tensor3` for `3d data` and `T.tensor4` for `4d data`. Next add the layers

```python
from mozi.layers.linear import Linear
from mozi.layers.activation import RELU, Softmax
from mozi.layers.noise import Dropout

model.add(Linear(prev_dim=28*28, this_dim=200))
model.add(RELU())
model.add(Linear(prev_dim=200, this_dim=100))
model.add(RELU())
model.add(Dropout(0.5))
model.add(Linear(prev_dim=100, this_dim=10))
model.add(Softmax())
```
To train the model, first build a dataset and a learning method, here we use the mnist dataset and SGD
```python
from mozi.datasets.mnist import Mnist
from mozi.learning_method import SGD

data = Mnist(batch_size=64, train_valid_test_ratio=[5,1,1])
learning_method = SGD(learning_rate=0.1, momentum=0.9, lr_decay_factor=0.9, decay_batch=10000)
```
Finally build a training object and put everything in to train the model
```python
from mozi.train_object import TrainObject
from mozi.cost import mse, error

train_object = TrainObject(model = model,
                           log = None,
                           dataset = data,
                           train_cost = mse,
                           valid_cost = error,
                           learning_method = learning_method,
                           stop_criteria = {'max_epoch' : 10,
                                            'epoch_look_back' : 5,
                                            'percent_decrease' : 0.01}
                           )
train_object.setup()
train_object.run()
```
#### Stopping Criteria
```python
stop_criteria = {'max_epoch' : 10,
                 'epoch_look_back' : 5,
                 'percent_decrease' : 0.01}
```
The stopping criteria here means the training will stop if training has reach 'max_epoch' of 10 or the validation error does not decrease by at least 1% in the past 5 epoch.
#### Test Model
And that's it! Once the training is done, to test the model, it's as simple as calling the forward propagation `fprop(X)` in model
```python
import numpy as np

ypred = model.fprop(data.get_test().X)
ypred = np.argmax(ypred, axis=1)
y = np.argmax(data.get_test().y, axis=1)
accuracy = np.equal(ypred, y).astype('f4').sum() / len(y)
print 'test accuracy:', accuracy
```
---
### More Examples
Mozi can be used to build effectively any kind of architecture. Below is another few examples
* [**Convolution Neural Network**](doc/cnn.md)
* [**Denoising Autoencoder**](doc/dae.md)
* [**Variational Autoencoder**](doc/vae.md)
* [**Alexnet**](example/voc_alexnet.py)

---
### Layer Template
To build a layer for Mozi, the layer has to implement the template
```python
class Template(object):
    """
    DESCRIPTION:
        The interface to be implemented by any layer.
    """
    def __init__(self):
        self.params = [] # all params that needs to be updated by training go into the list

    def _test_fprop(self, state_below):
        # the testing track whereby no params update is performed after data flows through this track
        raise NotImplementedError()

    def _train_fprop(self, state_below):
        # the training track whereby params is updated every time data flows through this track
        raise NotImplementedError()

    def _layer_stats(self, state_below, layer_output):
        # calculate everything you want to know about the layer, the input, the output,
        # the weight and put in the return list in the format [('W max', T.max(W)), ('W min', T.min(W))].
        # This method provides a peek into the layer and is useful for debugging.
        return []
```
Each layer provides two tracks: training track and testing track. During training, the model will call `_train_fprop` in every layer and the output from the model will be used to update the params in `self.params` in each layer. During testing, `_test_fprop` is called in every layer and the output is used to evaluate the model and to judge if model should stop training based on the stopping criteria set in the `TrainObject`. We can also peek into each layer by putting whatever we want to know about the layer into `_layer_stats`. For example, if we want to know what is the maximum weight in a layer, we can compute `T.max(W)` and return `[('W max', T.max(W))]` from `_layer_stats`, so that after every epoch, `'W max'` will be calculated for that layer and output to screen.

---
### Data Interface
Mozi provides two data interface, one is for dataset small enough to fit all into the memory [(SingleBlock)](mozi/datasets/dataset.py#L82), another is for large datasets which cannot be fit into memory in one go and has to be broken up into blocks and load into training one block at a time [(DataBlocks)](mozi/datasets/dataset.py#L171).  
Check out the [Mnist](mozi/datasets/mnist.py) or [Cifar10](mozi/datasets/cifar10.py) examples on how to build a dataset.

#### Using SingleBlock Directly
Besides subclassing `SingleBlock` or `DataBlocks` to create dataset like Mnist and Cifar10 example, we can use `SingleBlock` or `DataBlocks` directly to build a dataset in as simple as
```python
from mozi.datasets.dataset import SingleBlock
import numpy as np
X = np.random.rand(1000, 5)
y = np.random.rand(1000, 3)
data = SingleBlock(X=X, y=y, batch_size=100, train_valid_test_ratio=[3,1,1])
train_object = TrainObject(dataset = data)
```

---
### Logging
Mozi provides a logging module for automatic saving of best model and logging the errors for each epoch.


```python
from mozi.log import Log

log = Log(experiment_name = 'MLP',
          description = 'This is a tutorial',
          save_outputs = True, # log all the outputs from the screen
          save_model = True, # save the best model
          save_epoch_error = True, # log error at every epoch
          save_to_database = {'name': 'Example.db',
                              'records': {'Batch_Size': data.batch_size,
                                          'Learning_Rate': learning_method.learning_rate,
                                          'Momentum': learning_method.momentum}}
         ) # end log
```
The log module allows logging of outputs from screen, saving best model and epoch-errors. It also allows recording of hyperparameters to the database using the `save_to_database` argument, the `save_to_database` argument takes in a dictionary that contains two fields `'name'` and `'records'`. `'name'` indicates the name of the database to save the recording table. The name of the recording table will follow the experiment name under argument `experiment_name`. The `'records'` field takes in a dictionary of unrestricted number of hyperparameters that we want to record. The `'records'` only accepts primitive data types (str, int, float).
Once log object is built, it can be passed into `TrainObject` as
```python
TrainObject(log = log)
```
<!--### More Features-->
<!--##### Data Iterators-->
<!--##### Customized Learning Method-->
<!--##### Customized Weight Initialization-->
---
### Why Mozi?
[Mozi](https://en.wikiquote.org/wiki/Mozi) (墨子) (470 B.C - 391 B.C) is a Chinese philosopher during warring states period (春秋戰國), his philosophy advocates peace, simplicity, universal love and pragmatism.

---
### Licence
MIT Licence
