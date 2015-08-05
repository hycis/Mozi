
Convolution Neural Network
=====
You can try the Convolution Neural Network [Example](../example/cifar10_cnn.py) running on Cifar10. Here we build four convolution layers with two fully-connected layers
```python
from mozi.model import Sequential
from mozi.layers.linear import Linear
from mozi.layers.noise import Dropout
from mozi.layers.activation import *
from mozi.layers.convolution import *
from mozi.layers.misc import Flatten
import theano.tensor as T

model = Sequential(input_var=T.tensor4())
model.add(Convolution2D(input_channels=3, filters=32, kernel_size=(3,3), stride=(1,1), border_mode='full'))
model.add(RELU())
model.add(Convolution2D(input_channels=32, filters=32, kernel_size=(3,3), stride=(1,1)))
model.add(RELU())
model.add(Pooling2D(poolsize=(2, 2), mode='max'))
model.add(Dropout(0.25))

model.add(Convolution2D(input_channels=32, filters=64, kernel_size=(3,3), stride=(1,1), border_mode='full'))
model.add(RELU())
model.add(Convolution2D(input_channels=64, filters=64, kernel_size=(3,3), stride=(1,1),))
model.add(RELU())
model.add(Pooling2D(poolsize=(2, 2), mode='max'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Linear(64*8*8, 512))
model.add(RELU())
model.add(Dropout(0.5))

model.add(Linear(512, 10))
model.add(Softmax())
```
Next build the `Cifar10` dataset, `LearningMethod` and put everything in `TrainObject` and run
```python
from mozi.datasets.cifar10 import Cifar10
from mozi.train_object import TrainObject
from mozi.cost import error, entropy
from mozi.learning_method import SGD

data = Cifar10(batch_size=64, train_valid_test_ratio=[5,1,1])
learning_method = SGD(learning_rate=0.01, momentum=0.9,
                      lr_decay_factor=0.9, decay_batch=5000)
train_object = TrainObject(model = model,
                           log = None,
                           dataset = data,
                           train_cost = entropy,
                           valid_cost = error,
                           learning_method = learning_method,
                           stop_criteria = {'max_epoch' : 10,
                                            'epoch_look_back' : 5,
                                            'percent_decrease' : 0.01}
                           )
train_object.setup()
train_object.run()
```
