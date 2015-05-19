from jobman import DD, flatten

#####################[Two_Layers_No_Transpose]######################
####################################################################

config = DD({

        'module_name'                   : 'Two_Layers_No_Transpose',

        'model' : DD({
                'rand_seed'             : 4520
                }), # end mlp

        'log' : DD({
                'experiment_name'       : 'AE1107_No_Transpose_Scale_Warp_Blocks_2049_500_gpu_adagrad_dropout',
                'description'           : '',
                'save_outputs'          : True,
                'save_learning_rule'    : True,
                'save_model'            : True,
                'save_epoch_error'      : True,
                'save_to_database_name' : 'Laura5.db'
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
                'learning_rate'         : 0.9,
                'momentum'              : 0.01,

                # for AdaDelta
                'rho'                   : 0.95,
                'eps'                   : 1e-6,
                }), # end learning_method

        #===========================[ Dataset ]===========================#
        'dataset' : DD({
                # 'type'                  : 'Laura_Warp_Blocks_180',
                # 'type'                  : 'Laura_Cut_Warp_Blocks_300',
                # 'type'                  : 'Laura_Blocks_500',
                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',
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

                'batch_size'            : (50, 100, 150, 200),
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        # #============================[ Layers ]===========================#
        'num_layers' : 1,

        'hidden1' : DD({
                'name'                  : 'hidden1',
                'type'                  : 'Tanh',
                'dim'                   : 500,
                'dropout_below'         : 0.5,
                'layer_noise'           : None,
                # 'layer_noise'           : 'BlackOut',
                # 'layer_noise'           : 'Gaussian',
                # 'layer_noise'           : 'MaskOut',
                # 'layer_noise'           : 'BatchOut',
                }), # end hidden_layer


        'h1_mirror' : DD({
                'name'                  : 'h1_mirror',
                'type'                  : 'Tanh',
                # 'dim'                   : 2049, # dim = input.dim
                'dropout_below'         : 0.5,
                'layer_noise'           : None,
                # 'layer_noise'           : 'BlackOut',
                # 'layer_noise'           : 'Gaussian',
                # 'layer_noise'           : 'MaskOut',
                # 'layer_noise'           : 'BatchOut',
                }) # end output_layer


        }), # end autoencoder
