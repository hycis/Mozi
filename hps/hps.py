
from jobman import DD, expand, flatten

import smartNN.layer as layer

from smartNN.mlp import MLP
from smartNN.layer import RELU, Sigmoid, Softmax, Linear
from smartNN.datasets.mnist import Mnist
from smartNN.datasets.spec import P276
from smartNN.learning_rule import LearningRule
from smartNN.log import Log
from smartNN.train_object import TrainObject
from smartNN.cost import Cost
import smartNN.datasets.preprocessor as preproc

import os

class AE_HPS:

    def __init__(self, state):
        self.state = state


    def run(self):
        log = self.build_log()
        dataset = self.build_dataset()
        
        train = dataset.get_train()
        dataset.set_train(train.X, train.X)
    
        valid = dataset.get_valid()
        dataset.set_valid(valid.X, valid.X)
    
        test = dataset.get_test()
        dataset.set_test(test.X, test.X)
        
        learning_rule = self.build_learning_rule()
        mlp = self.build_mlp(dataset)
        train_obj = TrainObject(log = log, 
                                dataset = dataset, 
                                learning_rule = learning_rule, 
                                model = mlp)
        train_obj.run()
        
        
    def build_log(self):
        log = Log(experiment_id = self.state.log.experiment_id,
                description = self.state.log.description,
                save_outputs = self.state.log.save_outputs,
                save_hyperparams = self.state.log.save_hyperparams,
                save_model = self.state.log.save_model,
                send_to_database = self.state.log.send_to_database)
        return log
    
    def build_dataset(self):
        
        dataset = None
    
        preprocessor = None if self.state.dataset.preprocessor is None else \
                       getattr(preproc, self.state.dataset.preprocessor)()
        
        if self.state.dataset.type == 'Mnist':
            dataset = Mnist(preprocessor = preprocessor,
                            binarize = self.state.dataset.binarize,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            train_ratio = self.state.dataset.train_ratio,
                            valid_ratio = self.state.dataset.valid_ratio,
                            iter_class = self.state.dataset.iter_class)
                            
        elif self.state.dataset.type == 'P276':
            dataset = P276(preprocessor = preprocessor,
                            feature_size = self.state.dataset.feature_size,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            train_ratio = self.state.dataset.train_ratio,
                            valid_ratio = self.state.dataset.valid_ratio,
                            test_ratio = self.state.dataset.test_ratio,
                            iter_class = self.state.dataset.iter_class)
        return dataset
    
    def build_learning_rule(self):
        learning_rule = LearningRule(max_col_norm = self.state.learning_rule.max_col_norm,
                                    learning_rate = self.state.learning_rule.learning_rate,
                                    momentum = self.state.learning_rule.momentum,
                                    momentum_type = self.state.learning_rule.momentum_type,
                                    weight_decay = self.state.learning_rule.weight_decay,
                                    cost = Cost(type = self.state.learning_rule.cost),
                                    stopping_criteria = {'max_epoch' : self.state.learning_rule.stopping_criteria.max_epoch,
                                                        'epoch_look_back' : self.state.learning_rule.stopping_criteria.epoch_look_back,
                                                        'cost' : Cost(type=self.state.learning_rule.stopping_criteria.cost),
                                                        'percent_decrease' : self.state.learning_rule.stopping_criteria.percent_decrease})
        return learning_rule
    
    def build_mlp(self, dataset):
    
        mlp = MLP(input_dim = dataset.feature_size())
        hidden_layer = getattr(layer, self.state.hidden_layer.type)(dim=self.state.hidden_layer.dim, 
                                                                    name=self.state.hidden_layer.name,
                                                                    dropout_below=self.state.hidden_layer.dropout_below)
        mlp.add_layer(hidden_layer)
        
        output_layer = getattr(layer, self.state.output_layer.type)(dim=dataset.target_size(), 
                                                                    name=self.state.output_layer.name,
                                                                    W=hidden_layer.W.T,
                                                                    dropout_below=self.state.output_layer.dropout_below)
        mlp.add_layer(output_layer)
        return mlp
             
                    