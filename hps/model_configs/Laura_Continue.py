from jobman import DD, flatten

##########################[Laura_Continue]########################
##################################################################

config = DD({

        'module_name'                   : 'Laura_Continue',

        'model' : DD({
                'rand_seed'             : None
                }), # end mlp

        'log' : DD({

                'experiment_name'       : 'AE0302_Scale_Warp_Blocks_2049_500_Clean_AdaGrad_20150304_0512_00344145_continue',



                'description'           : '',
                'save_outputs'          : True,
                'save_learning_rule'    : True,
                'save_model'            : True,
                'save_epoch_error'      : True,
                'save_to_database_name' : 'Laura13.db'
                }), # end log


        'learning_rule' : DD({
                'max_col_norm'          : 1,
                'L1_lambda'             : None,
                'L2_lambda'             : None,
                'cost'                  : 'mse',
                'stopping_criteria'     : DD({
                                            'max_epoch'         : 100,
                                            'epoch_look_back'   : 5,
                                            'cost'              : 'mse',
                                            'percent_decrease'  : 0.05
                                            }) # end stopping_criteria
                }), # end learning_rule


        'learning_method' : DD({
                # 'type'                  : 'SGD',
                'type'                  : 'AdaGrad',
                # 'type'                  : 'AdaDelta',

                # for SGD and AdaGrad
                'learning_rate'         : 0.01,
                'momentum'              : 0.01,

                # for AdaDelta
                'rho'                   : 0.95,
                'eps'                   : 1e-6,
                }), # end learning_method

        #===========================[ Dataset ]===========================#
        'dataset' : DD({
                # 'type'                  : 'Laura_Warp_Blocks_500',
                # 'type'                  : 'Laura_Blocks_500',
                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',
                # 'type'                  : 'Mnist_Blocks',
                # 'type'                  : 'Laura_Scale_Warp_Blocks_500_Tanh',
                # 'type'                  : 'Laura_Warp_Blocks_500_Tanh_Noisy_MaskOut',
                # 'type'                  : 'Laura_Warp_Blocks_500_Tanh_Noisy_Gaussian',

                'num_blocks'            : 20,
                'feature_size'          : 2049,
                'train_valid_test_ratio': [8, 1, 1],

                'dataset_noise'         : DD({
                                            # 'type'              : 'BlackOut',
                                            # 'type'              : 'MaskOut',
                                            # 'type'              : 'Gaussian',
                                            'type'              : None
                                            }),

                'preprocessor'          : DD({
                                            # 'type' : None,
                                            'type' : 'Scale',
                                            # 'type' : 'GCN',
                                            # 'type' : 'LogGCN',
                                            # 'type' : 'Standardize',

                                            # for Scale
                                            'global_max' : 89,
                                            'global_min' : -23,
                                            'buffer'     : 0.5,
                                            'scale_range': [-1, 1],
                                            }),

                'batch_size'            : 100,
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        # #============================[ Layers ]===========================#
        'fine_tuning_only'              : True,
        'hidden1' : DD({
                'name'                  : 'hidden1',
                'model'                 : 'AE0302_Scale_Warp_Blocks_2049_500_Clean_AdaGrad_20150304_0512_00344145',
                }), # end hidden_layer


        }) # end autoencoder
