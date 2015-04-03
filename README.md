Pynet
=====

Pynet is meant to be a simple and straight forward framework base on Theano, it aims to be a much cleaner and focused framework than pylearn2, and only aims to do the best for neural network computation. It is for anyone who wants to run large dataset on GPU with 10X speedup, and who has a tough time learning the much more bulky pylearn2.

Pynet has been used to reproduce many state-of-the-art results, such as dropout and maxout on mnist and cifar10. Currently it's a stable and fast.

The list functions that Pynet support includes:
1. Control over the noise added each layer
2. Allows dropout
3. Can choose SGD, AdaDelta, AdaGrad or implement your own customize learning method
4. Allow model saving, logging of outputs, saving of the error vs epochs.
5. Allow launching of many jobs to the cluster or on the single machine.
6. Collect results from all the jobs into a database for comparison


Pynet consists of the following modules

1. [TrainObject](doc/train_object.md)
2. [Model](doc/model.md)
3. [Dataset](doc/dataset.md)
4. [Learning Method](doc/learning_method.md)
5. [Layer](doc/layer.md)
6. [Learning Rule](doc/learning_rule.md)
7. [Log](doc/log.md)

__Get Started with Simple Example__

You can start with a [MLP example](doc/mlp_example.md)

Or start with a [AutoEncoder example](doc/ae_example.md)
