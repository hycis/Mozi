# -*- coding: utf-8 -*-
# import time, sys, cPickle, os, socket
# 
# from pylearn2.utils import serial
# from itertools import izip
# from pylearn2.utils import safe_zip
# from collections import OrderedDict
# from pylearn2.utils import safe_union
# 
# import numpy as np
# import scipy.sparse as spp
# import theano.sparse as S
# 
# from theano.gof.op import get_debug_values
# from theano.printing import Print
# from theano import function
# from theano import config
# from theano.sandbox.rng_mrg import MRG_RandomStreams
# from theano import tensor as T
# import theano
# 
# from pylearn2.linear.matrixmul import MatrixMul
# 
# from pylearn2.models.model import Model
# 
# from pylearn2.training_algorithms.sgd import SGD, MomentumAdjustor
# from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
# from pylearn2.train import Train
# from pylearn2.costs.cost import SumOfCosts, Cost, MethodCost
# from pylearn2.costs.mlp import WeightDecay, L1WeightDecay
# from pylearn2.models.mlp import MLP, ConvRectifiedLinear, \
#     RectifiedLinear, Softmax, Sigmoid, Linear, Tanh, max_pool_c01b, \
#     max_pool, Layer
# from pylearn2.models.maxout import Maxout, MaxoutConvC01B
# from pylearn2.monitor import Monitor
# from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace, Space
# from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
# from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
# from pylearn2.datasets import preprocessing as pp
# 
# from layers import NoisyRELU, GaussianRELU, My_Softmax, My_MLP, My_Tanh
# 
# from dataset import My_CIFAR10
# 
# from pylearn2_objects import *
# #from load_model import compute_nll
# 
# 
# from pylearn2.datasets.mnist import MNIST
# from pylearn2.datasets.svhn import SVHN

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
             
                    