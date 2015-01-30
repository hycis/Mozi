Pynet
=====

Pynet is meant to be a simple and straight forward framework base on Theano, it aims to be a much cleaner and focused framework than pylearn2, and only aims to do the best for neural network computation. It is for anyone who wants to run large dataset on GPU with 10X speedup, and who has a tough time learning the much more bulky pylearn2.

Pynet has been used to reproduce many state-of-the-art results, such as dropout and maxout on mnist And it's a stable and fast.

Pynet consists of the following modules

1. TrainObject
2. Model
3. Layer
4. Dataset
5. Log

_Example_

To start with a simple MLP example click [here](doc/mlp_example.md)

Let's start with a simple example for building a neural network from [mlp_example.py](example/model_config.py)
```python

mlp = MLP(input_dim = data.feature_size())

data = Mnist(preprocessor=None, train_valid_test_ratio=[5,1,1])
learning_method = SGD(learning_rate=0.001, momentum=0.9)

mlp.add_layer(Sigmoid(dim=400, name='h1_layer', W=None, b=None, dropout_below=None))
mlp.add_layer(Softmax(dim=data.target_size(), name='output_layer', W=None, b=None, dropout_below=None))

learning_rule = LearningRule(max_col_norm = 10,
                            L1_lambda = None,
                            L2_lambda = None,
                            training_cost = Cost(type='mse'),
                            learning_rate_decay_factor = None,
                            stopping_criteria = {'max_epoch' : 30,
                                                'epoch_look_back' : 10,
                                                'cost' : Cost(type='error'),
                                                'percent_decrease' : 0.01}
                            )

log = Log(experiment_name = 'mnist_example',
            description = 'This is tutorial example',
            save_outputs = False,
            save_learning_rule = False,
            save_model = False,
            save_epoch_error = False,
            save_to_database = {'name': 'Example.db',
            'records' : {'Dataset' : data.__class__.__name__,
            'max_col_norm'     : learning_rule.max_col_norm,
            'Weight_Init_Seed' : mlp.rand_seed,
            'Dropout_Below'    : str([layer.dropout_below for layer in mlp.layers]),
            'Batch_Size'       : data.batch_size,
            'Layer_Dim'        : str([layer.dim for layer in mlp.layers]),
            'Layer_Types'      : str([layer.__class__.__name__ for layer in mlp.layers]),
            'Preprocessor'     : data.preprocessor.__class__.__name__,
            'Learning_Rate'    : learning_method.learning_rate,
            'Momentum'         : learning_method.momentum,
            'Training_Cost'    : learning_rule.cost.type,
            'Stopping_Cost'    : learning_rule.stopping_criteria['cost'].type}}
            ) # end log

train_object = TrainObject(model = mlp,
dataset = data,
learning_rule = learning_rule,
learning_method = learning_method)
train_object.run()
```




Let's start with a simple example for building a denoising autoencoder


Dataset Template
    - preprocessor
    - dataset noise
    - iterator

Layer Template
    - layer noise


Learning Method Template


Cost Functions
