
from jobman import DD, flatten


#############################[Laura]##############################
##################################################################

config = DD({

        'module_name'                   : 'Laura',

        'model' : DD({
                # 'rand_seed'             : 4520,
                'rand_seed'             : None,
                # 'rand_seed'             : 2137
                }), # end mlp

        'log' : DD({
                # 'experiment_name'       : 'testing_blackout',
                # 'experiment_name'       : 'AE0910_Warp_Blocks_2049_500_tanh_gpu_blockout_more_no_filter_latest',
                # 'experiment_name'       : 'AE0829_Warp_Standardize_GCN_Blocks_2049_500_tanh_gpu',
                # 'experiment_name'       : 'AE0912_Blocks_2049_500_tanh_gpu_clean',
                # 'experiment_name'       : 'AE0829_Standardize_GCN_Blocks_2049_500_tanh_gpu',
                # 'experiment_name'       : 'AE0901_Warp_Blocks_500_180_tanh_gpu',

                # 'experiment_name'       : 'AE1016_Warp_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                # 'experiment_name'       : 'AE1018_Warp_Blocks_2049_500_tanh_tanh_gpu_blackout', #helios

                # 'experiment_name'       : 'AE0919_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                # 'experiment_name'       : 'AE0918_Blocks_180_120_tanh_tanh_gpu_clean', #helios

                # 'experiment_name'       : 'AE0916_Blocks_180_120_tanh_tanh_gpu_output_sig_dropout',
                # 'experiment_name'       : 'AE0916_Blocks_180_120_tanh_tanh_gpu_output_sig_clean',

                # 'experiment_name'       : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                # 'experiment_name'       : 'AE1210_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_sgd_maskout', #helios

                'experiment_name'       : 'AE0314_Scale_Warp_Blocks_2049_120_Clean',

                # 'experiment_name'       : 'AE1216_Transfactor_blocks_150_50small',

                'description'           : 'scale_buffer=0.5',
                'save_outputs'          : True,
                'save_learning_rule'    : True,
                'save_model'            : True,
                'save_epoch_error'      : True,
                'save_to_database_name' : 'Laura13.db'
                # 'save_to_database_name' : 'transfactor.db',
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
                'type'                  : 'SGD',
                # 'type'                  : 'AdaGrad',
                # 'type'                  : 'AdaDelta',

                ###[ For SGD and AdaGrad ]###
                # 'learning_rate'         : 0.001,
                'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),

                # 'momentum'              : 0.5,
                # 'momentum'              : 0.,
                'momentum'              : (1e-2, 1e-1, 0.5, 0.9),

                ###[ For AdaDelta ]###
                'rho'                   : ((0.9, 0.99),float),
                'eps'                   : ((1e0, 1e-8),float),
                }), # end learning_method

        #===========================[ Dataset ]===========================#
        'dataset' : DD({
                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',

                # 'type'                  : 'Laura_Warp_Blocks_1000_Tanh',
                # 'type'                 : 'Laura_Warp_Blocks_180_Tanh_Dropout',
                # 'type'                  : 'Laura_Cut_Warp_Blocks_300',
                # 'type'                  : 'Laura_Blocks_180_Tanh_Tanh',
                # 'type'                  : 'Laura_Blocks_180_Tanh_Tanh_Dropout',
                # 'type'                  : 'Laura_Blocks_500_Tanh_Sigmoid',
                # 'type'                  : 'Laura_Blocks_500',

                # 'type'                  : 'Laura_Warp_Standardize_Blocks',
                # 'type'                  : 'Laura_Standardize_Blocks',

                # 'type'                  : 'Laura_Scale_Warp_Blocks_500_Tanh',
                # 'type'                  : 'Laura_Scale_Warp_Blocks_180_Tanh_Dropout',

                # 'type'                  : 'Laura_Warp_Blocks_180_Tanh_Blackout',

                # 'type'                  : 'Mnist',

                # 'type'                  : 'Laura_Warp_Blocks_180_Tanh_Noisy_MaskOut',
                # 'type'                  : 'TransFactor_AE',
                # 'type'                  : 'TransFactor_Blocks150',

                'num_blocks'            : 20,
                'feature_size'          : 2049,
                'train_valid_test_ratio': [8, 1, 1],

                'dataset_noise'         : DD({
                                            'type'              : None
                                            # 'type'              : 'BlackOut',
                                            # 'type'              : 'MaskOut',
                                            # 'type'              : 'Gaussian',
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
                # 'batch_size'            : 50,
                'batch_size'            : (50, 100, 150, 200),
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        #============================[ Layers ]===========================#
        'num_layers' : 3,

        'hidden1' : DD({
                'name'                  : 'hidden1',
                'type'                  : 'Tanh',
                # 'type'                  : 'SoftRELU',
                'dim'                   : 500,

                'dropout_below'         : None,
                # 'dropout_below'         : (0.3, 0.4, 0.5),
                # 'dropout_below'         : 0.5,

                'layer_noise'           : DD({
                                            'type'      : None,
                                            # 'type'      : 'BlackOut',
                                            # 'type'      : 'Gaussian',
                                            # 'type'      : 'MaskOut',
                                            # 'type'      : 'BatchOut',

                                            # for BlackOut, MaskOut and BatchOut
                                            # 'ratio'     : (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                                            'ratio'     : 0.5,

                                            # for Gaussian
                                            # 'std'       : (0.001, 0.005, 0.01, 0.015, 0.02),
                                            'std'       : (0.005, 0.01, 0.02, 0.03, 0.04),
                                            # 'std'       : 0.001,
                                            'mean'      : 0,
                                            })

                }), # end hidden_layer

        'hidden2' : DD({
                'name'                  : 'hidden2',
                'type'                  : 'Tanh',
                'dim'                   : 180,
                'dropout_below'         : None,

                'layer_noise'           : DD({
                                            'type'      : None,
                                            # 'type'      : 'BlackOut',
                                            # 'type'      : 'Gaussian',
                                            # 'type'      : 'MaskOut',
                                            # 'type'      : 'BatchOut',

                                            # for BlackOut, MaskOut and BatchOut
                                            # 'ratio'     : (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                                            'ratio'     : 0.5,

                                            # for Gaussian
                                            # 'std'       : (0.001, 0.005, 0.01, 0.015, 0.02),
                                            'std'       : (0.005, 0.01, 0.02, 0.03, 0.04),
                                            # 'std'       : 0.001,
                                            'mean'      : 0,
                                            })
                }), # end hidden_layer

        'hidden3' : DD({
                'name'                  : 'hidden3',
                'type'                  : 'Tanh',
                'dim'                   : 120,
                'dropout_below'         : None,

                'layer_noise'           : DD({
                                            'type'      : None,
                                            # 'type'      : 'BlackOut',
                                            # 'type'      : 'Gaussian',
                                            # 'type'      : 'MaskOut',
                                            # 'type'      : 'BatchOut',

                                            # for BlackOut, MaskOut and BatchOut
                                            # 'ratio'     : (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
                                            'ratio'     : 0.5,

                                            # for Gaussian
                                            # 'std'       : (0.001, 0.005, 0.01, 0.015, 0.02),
                                            'std'       : (0.005, 0.01, 0.02, 0.03, 0.04),
                                            # 'std'       : 0.001,
                                            'mean'      : 0,
                                            })
                }), # end hidden_layer

        'h3_mirror' : DD({
                'name'                  : 'h3_mirror',
                'type'                  : 'Tanh',
                # 'dim'                   : 2049, # dim = hidden2.dim
                'dropout_below'         : None,

                }), # end output_layer

        'h2_mirror' : DD({
                'name'                  : 'h2_mirror',
                'type'                  : 'Tanh',
                # 'dim'                   : 2049, # dim = hidden2.dim
                'dropout_below'         : None,

                }), # end output_layer

        'h1_mirror' : DD({
                'name'                  : 'h1_mirror',
                'type'                  : 'Tanh',
                # 'dim'                   : 2049, # dim = input.dim

                'dropout_below'         : None,
                # 'dropout_below'         : 0.5,

                }) # end output_layer

    }) # end Laura
