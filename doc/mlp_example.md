MLP Example
===========

To build an mlp, we need to put together the following modules

1. TrainObject
2. [MLP](model.md)
3. [Dataset](dataset.md)
4. [Learning Method](learning_method.md): Can be SGD, AdaDelta or AdaGrad
5. [Layer](layer.md)
6. [Learning Rule](learning_rule.md)
7. [Log](log.md): for logging all the information and saving the model

below is an example code for building an mlp from [mlp_example.py](../example/mlp_example.py)
```python

# build dataset
data = Mnist(preprocessor=None, train_valid_test_ratio=[5,1,1])

# build mlp
mlp = MLP(input_dim = data.feature_size())
mlp.add_layer(Sigmoid(dim=1000, name='h1_layer', W=None, b=None, dropout_below=None))
mlp.add_layer(Softmax(dim=data.target_size(), name='output_layer', W=None, b=None, dropout_below=None))

# build learning method
learning_method = SGD(learning_rate=0.1, momentum=0.9)

# set the learning rules
learning_rule = LearningRule(max_col_norm = 10,
                            L1_lambda = None,
                            L2_lambda = None,
                            training_cost = Cost(type='mse'),
                            learning_rate_decay_factor = None,
                            stopping_criteria = {'max_epoch' : 300,
                                                  'epoch_look_back' : 10,
                                                  'cost' : Cost(type='error'),
                                                  'percent_decrease' : 0.01}
                            )

# (optional) build the logging object
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

# put everything into the train object
train_object = TrainObject(model = mlp,
                           dataset = data,
                           learning_rule = learning_rule,
                           learning_method = learning_method
                           log = log)
# finally run the code
train_object.run()
```
