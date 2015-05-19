from jobman import DD, flatten

##############################[ I2R ]#############################
##################################################################

config = DD({

    'module_name'                   : 'I2R',

    'model' : DD({
            'rand_seed'             : None
            }), # end mlp

    'log' : DD({
            'experiment_name'       : 'i2r0213_noisy_posterior_1layer_30blk',
            'description'           : '',
            'save_outputs'          : False,
            'save_learning_rule'    : False,
            'save_model'            : False,
            'save_epoch_error'      : False,
            'save_to_database_name' : "i2r.db"
            }), # end log

    'learning_method' : DD({
            'type'                  : 'SGD',
            # 'type'                  : 'AdaGrad',
            # 'type'                  : 'AdaDelta',

            ###[ For SGD and AdaGrad ]###
            # 'learning_rate'         : 0.5,
            # 'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
            # 'learning_rate'         : ((1e-3, 0.5), float),
            'learning_rate'         : 0.0305287335067987,
            'momentum'              : 0.9,
            # 'momentum'              : 0.,
            # 'momentum'              : (1e-2, 1e-1, 0.5, 0.9),

            # For AdaDelta
            'rho'                   : ((0.90, 0.99), float),
            'eps'                   : (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
            }),

    'learning_rule' : DD({
            'max_col_norm'          : 1,
            'L1_lambda'             : None,
            'L2_lambda'             : 0.0001,
            'cost'                  : 'mse',
            'stopping_criteria'     : DD({
                                        'max_epoch'         : 100,
                                        'epoch_look_back'   : 5,
                                        'cost'              : 'error',
                                        'percent_decrease'  : 0.05
                                        }) # end stopping_criteria
            }), # end learning_rule

    'dataset' : DD({

            'type'                  : 'I2R_Posterior_Blocks_ClnDNN_NoisyFeat',
            # 'type'                  : 'I2R_Posterior_Blocks_ClnDNN_CleanFeat',
            # 'type'                  : 'I2R_Posterior_NoisyFeat_Sample',
            # 'type'                  : 'I2R_Posterior_Gaussian_Noisy_Sample',

            'train_valid_test_ratio': [5, 1, 1],
            'feature_size'          : 1998,
            'target_size'           : 1998,
            'one_hot'               : True,
            'num_blocks'            : 30,

            'dataset_noise'         : DD({
                                        # 'type'              : 'BlackOut',
                                        # 'type'              : 'MaskOut',
                                        # 'type'              : 'Gaussian',
                                        'type'              : None

                                        }),

            'preprocessor'          : DD({
                                        # 'type' : None,
                                        # 'type' : 'Scale',
                                        # 'type' : 'GCN',
                                        # 'type' : 'LogGCN',
                                        'type' : 'Standardize',
                                        # 'type'  : 'Log',

                                        # for Scale
                                        'global_max' : 1.0,
                                        'global_min' : 0,
                                        'buffer'     : 0.,
                                        'scale_range': [0.5, 1.],
                                        }),

            # 'batch_size'            : (50, 100, 150, 200),
            'batch_size'            : 100,
            'num_batches'           : None,
            'iter_class'            : 'SequentialSubsetIterator',
            'rng'                   : None
            }), # end dataset

    #============================[ Layers ]===========================#
    'hidden1' : DD({
            'name'                  : 'hidden1',
            'type'                  : 'RELU',
            'dim'                   : 3000,

            # 'dropout_below'         : (0.05, 0.1, 0.15, 0.2)
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
            }), # end hidden_layer

    'hidden2' : DD({
            'name'                  : 'hidden2',
            'type'                  : 'RELU',
            'dim'                   : 3000,

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
            'type'                  : 'Softmax',
            'dim'                   : 1998,

            # 'dropout_below'         : 0.5,
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
            }) # end output_layer
    })
