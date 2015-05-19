from jobman import DD, flatten

##############################[ NN ]##############################
##################################################################

config = DD({

    'module_name'                   : 'NN',

    'model' : DD({
            'rand_seed'             : None
            }), # end mlp

    'log' : DD({
            'experiment_name'       : 'mlp_dropout',
            'description'           : '',
            'save_outputs'          : False,
            'save_learning_rule'    : False,
            'save_model'            : False,
            'save_epoch_error'      : False,
            'save_to_database_name' : "mnist_model.db"
            }), # end log

    'learning_method' : DD({
            # 'type'                  : 'SGD',
            # 'type'                  : 'AdaGrad',
            'type'                  : 'AdaDelta',

            ###[ For SGD and AdaGrad ]###
            # 'learning_rate'         : 0.001,
            # 'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
            'learning_rate'         : 0.1,
            'momentum'              : 0.5,
            # 'momentum'              : 0.,
            # 'momentum'              : (1e-2, 1e-1, 0.5, 0.9),

            # For AdaDelta
            # 'rho'                   : ((0.90, 0.99), float),
            'rho'                   : 0.95,
            # 'eps'                   : (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
            'eps'                   : 1e-6,
            }),

    'learning_rule' : DD({
            'max_col_norm'          : None,
            'L1_lambda'             : None,
            'L2_lambda'             : 0.0001,
            'cost'                  : 'entropy',
            'stopping_criteria'     : DD({
                                        'max_epoch'         : 10,
                                        'epoch_look_back'   : 5,
                                        'cost'              : 'error',
                                        'percent_decrease'  : 0.05
                                        }) # end stopping_criteria
            }), # end learning_rule

    'dataset' : DD({

            'type'                  : 'Mnist',
            'train_valid_test_ratio': [5, 1, 1],
            'feature_size'          : 784,
            'target_size'           : 10,


            'dataset_noise'         : DD({
                                        # 'type'              : 'BlackOut',
                                        # 'type'              : 'MaskOut',
                                        # 'type'              : 'Gaussian',
                                        'type'              : None
                                        }),

            'preprocessor'          : DD({
                                        'type' : None,
                                        # 'type' : 'Scale',
                                        # 'type' : 'GCN',
                                        # 'type' : 'LogGCN',
                                        # 'type' : 'Standardize',

                                        # for Scale
                                        'global_max' : 4.0,
                                        'global_min' : 0.,
                                        'buffer'     : 0.,
                                        'scale_range': [0., 1.],
                                        }),

            'batch_size'            : (50, 100, 150, 200),
            # 'batch_size'            : 20,
            'num_batches'           : None,
            'iter_class'            : 'SequentialSubsetIterator',
            'rng'                   : None
            }), # end dataset

    #============================[ Layers ]===========================#
    'hidden1' : DD({
            'name'                  : 'hidden1',
            'type'                  : 'Tanh',
            'dim'                   : 500,

            # 'dropout_below'         : (0.05, 0.1, 0.15, 0.2)
            # 'dropout_below'         : (0, 0.5),
            'dropout_below'         : None,

            'layer_noise'           : DD({
                                        'type'      : None,
                                        # 'type'      : 'BlackOut',
                                        # 'type'      : 'Gaussian',
                                        # 'type'      : 'MaskOut',
                                        # 'type'      : 'BatchOut',

                                        # for BlackOut, MaskOut and BatchOut
                                        'ratio'     : 0.5,

                                        # for Gaussian
                                        'std'       : 0.1,
                                        'mean'      : 0,
                                        })
            }), # end hidden_layer

    'output' : DD({
            'name'                  : 'output',
            'type'                  : 'Sigmoid',
            'dim'                   : 10,

            # 'dropout_below'         : 0.5,
            'dropout_below'         : None,

            'layer_noise'           : DD({
                                        'type'      : None,
                                        # 'type'      : 'BlackOut',
                                        # 'type'      : 'Gaussian',
                                        # 'type'      : 'MaskOut',
                                        # 'type'      : 'BatchOut',

                                        # for BlackOut, MaskOut and BatchOut
                                        'ratio'     : 0.5,

                                        # for Gaussian
                                        'std'       : 0.1,
                                        'mean'      : 0,
                                        })
            }) # end output_layer
    })
