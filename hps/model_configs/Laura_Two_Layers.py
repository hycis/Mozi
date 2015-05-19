from jobman import DD, flatten

########################[Laura_Two_Layers]########################
##################################################################

config = DD({

        'module_name'                   : 'Laura_Two_Layers',

        'model' : DD({
                'rand_seed'             : None
                }), # end mlp

        'log' : DD({
                # 'experiment_name'       : 'AE1214_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_maskout',
                'experiment_name'       : 'AE0306_Scale_Warp_Blocks_2Layers_finetune_2049_1000_180',
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
                'type'                  : 'SGD',
                # 'type'                  : 'AdaGrad',
                # 'type'                  : 'AdaDelta',

                # for SGD and AdaGrad
                'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                'momentum'              : (1e-2, 1e-1, 0.5, 0.9),

                # for AdaDelta
                'rho'                   : 0.95,
                'eps'                   : 1e-6,
                }), # end learning_method

        #===========================[ Dataset ]===========================#
        'dataset' : DD({

                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',
                # 'type'                  : 'TransFactor_Blocks',

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

                'batch_size'            : (50, 100, 150, 200),
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        # #============================[ Layers ]===========================#

        'hidden1' : DD({
                'name'                  : 'hidden1',

                # 'model'                 : 'AE0911_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140912_2337_04263067',
                # 'model'                 : 'AE1112_Scale_Warp_Blocks_2Layers_finetune_2049_180_tanh_tanh_gpu_clean_20141112_2145_06823495',
                # 'model'                 : 'AE1121_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_gaussian_continue_20141126_1543_50554671',
                # 'model'                 : 'AE1122_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_maskout_20141128_1421_47179280',
                # 'model'                 : 'AE1210_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_maskout_20141210_1728_15311837',
                # 'model'                 : 'AE0302_Scale_Warp_Blocks_2049_300_Clean_No_Pretrain_20150302_2336_46071497',
                'model'                 : 'AE0302_Scale_Warp_Blocks_2049_1000_Clean_No_Pretrain_20150302_1234_10065582',
                'dropout_below'         : None,
                # 'dropout_below'         : (0.1, 0.2, 0.3, 0.4, 0.5),
                # 'dropout_below'         : 0.1,
                }), # end hidden_layer

        'hidden2' : DD({
                'name'                  : 'hidden2',

                # 'model'                 : 'AE1001_Warp_Blocks_500_120_tanh_tanh_gpu_clean_20141003_0113_02206401',
                # 'model'                 : 'AE1115_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_sgd_clean_20141119_1327_11490503',
                # 'model'                 : 'AE1127_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_gaussian_20141127_1313_31905279',
                # 'model'                 : 'AE1201_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_clean_20141202_2352_57643114',
                # 'model'                 : 'AE1210_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_maskout_20141212_2056_15976132',
                # 'model'                 : 'AE0302_Scale_Warp_Blocks_300_180_Clean_No_Pretrain_20150304_0436_47181007',
                'model'                 : 'AE0302_Scale_Warp_Blocks_1000_180_Clean_20150304_0511_52424408',
                'dropout_below'         : None,
                })
        }) # end autoencoder
