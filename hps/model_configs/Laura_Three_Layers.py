from jobman import DD, flatten

########################[Laura_Three_Layers]########################
####################################################################

config = DD({

        'module_name'                   : 'Laura_Three_Layers',

        'fine_tuning_only'              : False,

        'model' : DD({
                'rand_seed'             : None
                }), # end mlp

        'log' : DD({

                # 'experiment_name'       : 'AE0917_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean',
                # 'experiment_name'       : 'AE0919_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy',

                # 'experiment_name'       : 'AE0917_Blocks_3layers_finetune_2049_120_tanh_sigmoid_gpu_clean',
                # 'experiment_name'       : 'AE0917_Blocks_3layers_finetune_2049_120_tanh_sigmoid_gpu_noisy',

                # 'experiment_name'       : 'AE0917_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_clean',
                # 'experiment_name'       : 'AE0919_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_noisy',

                # 'experiment_name'       : 'AE1002_Scale_Warp_Blocks_3Layers_finetune_2049_120_tanh_tanh_gpu_noisy',
                # 'experiment_name'       : 'AE1002_Scale_Warp_Blocks_3Layers_finetune_2049_120_tanh_tanh_gpu_clean',

                'experiment_name'       : 'AE1213_Scale_Laura_Warp_Blocks_3layers_finetune_2049_120_tanh_tanh_gpu_maskout',

                'description'           : '',
                'save_outputs'          : True,
                'save_learning_rule'    : True,
                'save_model'            : True,
                'save_epoch_error'      : True,
                'save_to_database_name' : 'Laura12.db'
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
                # 'learning_rate'         : 0.001,
                'momentum'              : (1e-2, 1e-1, 0.5, 0.9),
                # 'momentum'              : 0.1,
                # 'momentum'              : 0.5,

                # for AdaDelta
                'rho'                   : 0.95,
                'eps'                   : 1e-6,
                }), # end learning_method

        #===========================[ Dataset ]===========================#
        'dataset' : DD({

                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',

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
                                            'buffer'     : 0.9,
                                            'scale_range': [-1, 1],
                                            }),

                'batch_size'            : (50, 100, 150, 200),
                # 'batch_size'            : 50,
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        # #============================[ Layers ]===========================#

        'hidden1' : DD({
                'name'                  : 'hidden1',
                # 'model'                 : 'AE0911_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140912_2337_04263067',
                # 'model'                 : 'AE0916_Warp_Blocks_2049_500_tanh_tanh_gpu_dropout_20140916_1705_29139505',

                # 'model'                 :'AE0912_Blocks_2049_500_tanh_tanh_gpu_clean_20140914_1242_27372903',
                # 'model'                 : 'AE0915_Blocks_2049_500_tanh_tanh_gpu_Dropout_20140915_1900_37160748',

                # 'model'                 : 'AE1002_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_dropout_20141001_0321_33382955',
                # 'model'                 : 'AE0930_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140930_1345_29800576',

                # 'model'                 : 'AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_clean_continue_20141110_1235_21624029',
                # 'model'                 : 'AE1110_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_batchout_continue_20141111_0957_22484008',
                # 'model'                 : 'AE1121_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_gaussian_continue_20141126_1543_50554671',
                # 'model'                 : 'AE1122_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_maskout_20141128_1421_47179280',
                'model'                 : 'AE1210_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_sgd_maskout_20141210_1728_15311837',
                'dropout_below'         : None,
                # 'dropout_below'         : (0.1, 0.2, 0.3, 0.4, 0.5),
                # 'dropout_below'         : 0.1,

                }), # end hidden_layer

        'hidden2' : DD({
                'name'                  : 'hidden2',
                # 'model'                 : 'AE0914_Warp_Blocks_500_180_tanh_tanh_gpu_clean_20140915_0400_30113212',
                # 'model'                 : 'AE0918_Warp_Blocks_500_180_tanh_tanh_gpu_dropout_20140918_1125_23612485',

                # 'model'                 : 'AE0916_Blocks_500_180_tanh_tanh_gpu_clean_20140916_2255_06553688',
                # 'model'                 : 'AE0918_Blocks_500_180_tanh_tanh_gpu_dropout_20140918_0920_42738052',

                # 'model'                 : 'AE1001_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_dropout_20141001_2158_16765065',
                # 'model'                 : 'AE1001_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_clean_20141002_0348_53679208',

                # 'model'                 : 'AE1110_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_clean_20141111_2157_47387660',
                # 'model'                 : 'AE1111_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_batchout_continue_20141112_0844_45882544',
                # 'model'                 : 'AE1127_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_gaussian_20141127_1313_31905279',
                # 'model'                 : 'AE1201_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_clean_20141202_2352_57643114',
                'model'                 : 'AE1210_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_sgd_maskout_20141212_2056_15976132',

                'dropout_below'         : None,

                }), # end hidden_layer

        'hidden3' : DD({
                'name'                  : 'hidden3',
                # 'model'                 : 'AE0915_Warp_Blocks_180_120_tanh_gpu_dropout_clean_20140916_1028_26875210',
                # 'model'                 : 'AE0918_Warp_Blocks_180_120_tanh_tanh_gpu_dropout_20140919_1649_54631649',

                # 'model'                 : 'AE0914_Blocks_180_120_tanh_tanh_gpu_clean_20140918_0119_40376829',
                # 'model'                 : 'AE0919_Blocks_180_120_tanh_tanh_gpu_dropout_20140919_1345_22865393',

                # 'model'                 : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_dropout_20141002_1711_48207269',
                # 'model'                 : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_dropout_20141002_1457_08966968',
                # 'model'                 : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_clean_20141002_1713_16791523',

                # 'model'                 : 'AE1120_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_sgd_clean_20141122_0044_09351031',
                # 'model'                 : 'AE1121_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_sgd_batchout_20141122_0348_49379314',
                # 'model'                 : 'AE1127_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_sgd_gaussian_20141201_0345_39835964',
                # 'model'                 : 'AE1201_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_sgd_clean_20141204_0137_07827194',
                'model'                 : 'AE1210_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_sgd_maskout_20141213_1608_33432934',

                'dropout_below'         : None,

                }), # end hidden_layer


        }) # end autoencoder
