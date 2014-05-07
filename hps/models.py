
from jobman import DD, expand, flatten

import smartNN.layer as layer

from smartNN.model import *
from smartNN.layer import *
from smartNN.datasets.mnist import Mnist
import smartNN.datasets.spec as spec
from smartNN.learning_rule import LearningRule
from smartNN.log import Log
from smartNN.train_object import TrainObject
from smartNN.cost import Cost
import smartNN.datasets.preprocessor as preproc

import cPickle

import os

class AE:

    def __init__(self, state):
        self.state = state

    def run(self):
        log = self.build_log()
        dataset = self.build_dataset()
        

        
        learning_rule = self.build_learning_rule()
        model = self.build_model(dataset)
        train_obj = TrainObject(log = log, 
                                dataset = dataset, 
                                learning_rule = learning_rule, 
                                model = model)
        train_obj.run()
        
        
    def build_log(self, id=None):
        log = Log(experiment_name = id is not None and '%s_%s'%(self.state.log.experiment_name,id) \
                                    or self.state.log.experiment_name,
                description = self.state.log.description,
                save_outputs = self.state.log.save_outputs,
                save_hyperparams = self.state.log.save_hyperparams,
                save_model = self.state.log.save_model,
                send_to_database = self.state.log.send_to_database)
        return log
    
    def build_dataset(self, part=None):
        
        dataset = None
    
        preprocessor = None if self.state.dataset.preprocessor is None else \
                       getattr(preproc, self.state.dataset.preprocessor)()
        
        if self.state.dataset.type == 'Mnist':
            dataset = Mnist(train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,       
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)
                            
        elif self.state.dataset.type[:4] == 'P276':
            dataset = getattr(spec, self.state.dataset.type)(
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)
        
        elif self.state.dataset.type[:5] == 'Laura':
            assert split is not None, 'split name is required for Laura dataset'
            dataset = getattr(spec, self.state.dataset.type)(
                            part = part,
                            train_valid_test_ratio = self.state.dataset.train_valid_test_ratio,
                            preprocessor = preprocessor,
                            batch_size = self.state.dataset.batch_size,
                            num_batches = self.state.dataset.num_batches,
                            iter_class = self.state.dataset.iter_class,
                            rng = self.state.dataset.rng)
                            
        train = dataset.get_train()
        dataset.set_train(train.X, train.X)
    
        valid = dataset.get_valid()
        dataset.set_valid(valid.X, valid.X)
    
        test = dataset.get_test()
        dataset.set_test(test.X, test.X)
        
        return dataset
    
    def build_learning_rule(self):
        learning_rule = LearningRule(max_col_norm = self.state.learning_rule.max_col_norm,
                                    learning_rate = self.state.learning_rule.learning_rate,
                                    momentum = self.state.learning_rule.momentum,
                                    momentum_type = self.state.learning_rule.momentum_type,
                                    L1_lambda = self.state.learning_rule.L1_lambda,
                                    L2_lambda = self.state.learning_rule.L2_lambda,
                                    cost = Cost(type = self.state.learning_rule.cost),
                                    stopping_criteria = {'max_epoch' : self.state.learning_rule.stopping_criteria.max_epoch,
                                                        'epoch_look_back' : self.state.learning_rule.stopping_criteria.epoch_look_back,
                                                        'cost' : Cost(type=self.state.learning_rule.stopping_criteria.cost),
                                                        'percent_decrease' : self.state.learning_rule.stopping_criteria.percent_decrease})
        return learning_rule
    
    def build_model(self, dataset):
    
        model = AutoEncoder(input_dim = dataset.feature_size(), rand_seed=self.state.model.rand_seed)
        hidden_layer = getattr(layer, self.state.hidden_layer.type)(dim=self.state.hidden_layer.dim, 
                                                                    name=self.state.hidden_layer.name,
                                                                    dropout_below=self.state.hidden_layer.dropout_below)
        model.add_encode_layer(hidden_layer)
        output_layer = getattr(layer, self.state.output_layer.type)(dim=dataset.target_size(), 
                                                                    name=self.state.output_layer.name,
                                                                    W=hidden_layer.W.T,
                                                                    dropout_below=self.state.output_layer.dropout_below)
        model.add_decode_layer(output_layer)
        return model

class AE_Two_Layers(AE):

    def __init__(self, state):
        self.state = state
        
        
    def run(self):
        log = self.build_log()
        
        dataset = self.build_dataset()
        
        learning_rule = self.build_learning_rule()
        
        with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + 
                self.state.hidden1.model_name + '/model.pkl', 'rb') as f:
            print('unpickling model: ' + self.state.hidden1.model_name)
            h1 = cPickle.load(f)
        
        with open(os.environ['smartNN_SAVE_PATH'] + '/log/' + 
                self.state.hidden2.model_name + '/model.pkl', 'rb') as f:
            print('unpickling model: ' + self.state.hidden2.model_name)
            h2 = cPickle.load(f)
        
        model = AutoEncoder(input_dim = dataset.feature_size(), rand_seed=self.state.model.rand_seed)
        model.add_encode_layer(h1.layers[0])
        model.add_encode_layer(h2.layers[0])
        model.add_decode_layer(h2.layers[1])
        model.add_decode_layer(h1.layers[1])
        
        train_obj = TrainObject(log = log, 
                                dataset = dataset, 
                                learning_rule = learning_rule, 
                                model = model)
        train_obj.run()
        
class AE_Two_Layers_WO_Pretrain(AE):

    def __init__(self, state):
        self.state = state

    def run(self):
        log = self.build_log()
        dataset = self.build_dataset()
        learning_rule = self.build_learning_rule()
                
        model = self.build_model(dataset)
        
        train_obj = TrainObject(log = log, 
                                dataset = dataset, 
                                learning_rule = learning_rule, 
                                model = model)
        train_obj.run()
    
    def build_model(self, dataset):
    
        model = AutoEncoder(input_dim=dataset.feature_size(), rand_seed=self.state.model.rand_seed)
        hidden1 = getattr(layer, self.state.hidden1.type)(dim=self.state.hidden1.dim, 
                                                        name=self.state.hidden1.name,
                                                        dropout_below=self.state.hidden1.dropout_below)
        model.add_encode_layer(hidden1)
        
        hidden2 = getattr(layer, self.state.hidden2.type)(dim=self.state.hidden2.dim, 
                                                        name=self.state.hidden2.name,
                                                        dropout_below=self.state.hidden2.dropout_below)
        model.add_encode_layer(hidden2)
        
        hidden2_mirror = getattr(layer, self.state.hidden2.type)(dim=hidden1.dim,
                                                                name=self.state.hidden2.name + '_mirror',
                                                                W = hidden2.W.T)
        model.add_decode_layer(hidden2_mirror)
        
        hidden1_mirror = getattr(layer, self.state.hidden2.type)(dim=dataset.target_size(),
                                                                name=self.state.hidden1.name + '_mirror',
                                                                W = hidden1.W.T)
        model.add_decode_layer(hidden1_mirror)
        
        return model

class AE_Many_Splits(AE):

    def __init__(self, state):
        self.state = state

    
    def run(self):
        learning_rule = self.build_learning_rule()
                
        model = self.build_model(dataset)
        
        
        for part in self.state.parts:
        
            dataset = self.build_dataset(part)        
            log = self.build_log(part)

            train_obj = TrainObject(log = log, 
                                    dataset = dataset, 
                                    learning_rule = learning_rule, 
                                    model = model)
            train_obj.run()
    
    
    
    def build_log(self, split=None):
    
    
            
                    