from jobman import DD, flatten

model_config = DD({

        'AE_Testing' : DD({

            'model' : DD({
                    'rand_seed'             : None
                    }), # end mlp

            'log' : DD({
                    # 'experiment_name'         : 'AE_Testing_Mnist_784_500',
                    'experiment_name'       : 'AE_Testing_Mnist_500_100',
                    'description'           : '',
                    'save_outputs'          : True,
                    'save_learning_rule'      : True,
                    'save_model'            : True,
                    'save_to_database_name' : 'Database_Name.db'
                    }), # end log


            'learning_rule' : DD({
                    'max_col_norm'          : (1, 10, 50),
                    'learning_rate'         : (1e-4, 1e-3, 1e-2, 1e-1, 0.9),
                    'momentum'              : (1e-2, 1e-1, 0.5, 0.9),
                    'momentum_type'         : 'normal',
                    'L1_lambda'             : None,
                    'L2_lambda'             : None,
                    'cost'                  : 'mse',
                    'stopping_criteria'     : DD({
                                                'max_epoch'         : 100,
                                                'epoch_look_back'   : 10,
                                                'cost'              : 'mse',
                                                'percent_decrease'  : 0.05
                                                }) # end stopping_criteria
                    }), # end learning_rule

            'dataset' : DD({

                    'type'                  : 'Mnist_Blocks_500',
                    'train_valid_test_ratio': [8, 1, 1],
                    'feature_size'          : 500,
                    # 'preprocessor'          : None,
        #                     'preprocessor'          : 'Scale',
                    # 'preprocessor'          : 'GCN',
                            # 'preprocessor'          : 'LogGCN',
                    'preprocessor'          : 'Standardize',
                    'batch_size'            : 100,
                    'num_batches'           : None,
                    'iter_class'            : 'SequentialSubsetIterator',
                    'rng'                   : None
                    }), # end dataset

            #============================[ Layers ]===========================#
            'hidden1' : DD({
                    'name'                  : 'hidden1',
                    'type'                  : 'Sigmoid',
                    'dim'                   : 100,
                    # 'dropout_below'         : (0.05, 0.1, 0.15, 0.2)
                    'dropout_below'         : 0.5,
                    }), # end hidden_layer

            'h1_mirror' : DD({
                    'name'                  : 'h1_mirror',
                    'type'                  : 'Sigmoid',
                    # 'dim'                   : 2049, # dim = input.dim
                    'dropout_below'         : None,
                    }) # end output_layer
            }), # end autoencoder


        #############################[Mapping]############################
        ##################################################################

        'Laura_Mapping' : DD({

            'model' : DD({
                    'rand_seed'             : None
                    }), # end mlp

            'log' : DD({
                    'experiment_name'       : 'AE1001_Warp_Laura_Blocks_GCN_Mapping', #helios

                    'description'           : '',
                    'save_outputs'          : True,
                    'save_learning_rule'      : True,
                    'save_model'            : True,
                    'save_to_database_name' : 'Laura.db'
                    }), # end log


            'learning_rule' : DD({
                    'max_col_norm'          : (1, 10, 50),
                    # 'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                    'learning_rate'         : ((1e-8, 1e-3), float),
                    'momentum'              : (1e-3, 1e-2, 1e-1, 0.5, 0.9),
                    'momentum_type'         : 'normal',
                    'L1_lambda'             : None,
                    'L2_lambda'             : None,
                    'cost'                  : 'mse',
                    'stopping_criteria'     : DD({
                                                'max_epoch'         : 100,
                                                'epoch_look_back'   : 10,
                                                'cost'              : 'mse',
                                                'percent_decrease'  : 0.05
                                                }) # end stopping_criteria
                    }), # end learning_rule

            #===========================[ Dataset ]===========================#
            'dataset' : DD({
                    # 'type'                  : 'Laura_Blocks_GCN_Mapping',
                    'type'                  : 'Laura_Warp_Blocks_GCN_Mapping',

                    'feature_size'          : 2049,
                    'target_size'           : 1,
                    'train_valid_test_ratio': [8, 1, 1],

                    'preprocessor'          : 'GCN',

                    'batch_size'            : (50, 100, 150, 200),
                    'num_batches'           : None,
                    'iter_class'            : 'SequentialSubsetIterator',
                    'rng'                   : None
                    }), # end dataset

            #============================[ Layers ]===========================#
            'num_layers' : 1,

            'hidden1' : DD({
                    'name'                  : 'hidden1',
                    'type'                  : 'Tanh',
                    'dim'                   : 1000,
                    'dropout_below'         : None,


                    }), # end hidden_layer

            'hidden2' : DD({
                    'name'                  : 'hidden2',
                    'type'                  : 'Tanh',
                    'dim'                   : 500,
                    'dropout_below'         : None,
                    }), # end hidden_layer

            'output' : DD({
                    'name'                  : 'output',
                    'type'                  : 'Linear',
                    'dim'                   : 1,
                    'dropout_below'         : None,
                    }), # end hidden_layer

            }), # end Laura_Mapping

        #############################[Laura]##############################
        ##################################################################

        'Laura' : DD({

            'model' : DD({
                    'rand_seed'             : None
                    }), # end mlp

            'log' : DD({
                    # 'experiment_name'       : 'testing_bloout',
                    # 'experiment_name'       : 'AE0910_Warp_Blocks_2049_500_tanh_gpu_blockout_more_no_filter_latest',
                    # 'experiment_name'       : 'AE0829_Warp_Standardize_GCN_Blocks_2049_500_tanh_gpu',
                    # 'experiment_name'       : 'AE0912_Blocks_2049_500_tanh_gpu_clean',
                    # 'experiment_name'       : 'AE0829_Standardize_GCN_Blocks_2049_500_tanh_gpu',
                    # 'experiment_name'       : 'AE0901_Warp_Blocks_500_180_tanh_gpu',

                    # 'experiment_name'       : 'AE1016_Warp_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                    'experiment_name'       : 'AE1016_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_blackout', #helios
                    #
                    # 'experiment_name'       : 'AE0919_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                    # 'experiment_name'       : 'AE0918_Blocks_180_120_tanh_tanh_gpu_clean', #helios

                    # 'experiment_name'       : 'AE0916_Blocks_180_120_tanh_tanh_gpu_output_sig_dropout',
                    # 'experiment_name'       : 'AE0916_Blocks_180_120_tanh_tanh_gpu_output_sig_clean',

                    # 'experiment_name'       : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_dropout', #helios
                    # 'experiment_name'       : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_clean', #helios


                    'description'           : '',
                    'save_outputs'          : True,
                    'save_learning_rule'    : True,
                    'save_model'            : True,
                    'save_epoch_error'      : True,
                    'save_to_database_name' : 'Laura3.db'
                    }), # end log


            'learning_rule' : DD({
                    'max_col_norm'          : (1, 10, 50),
                    # 'learning_rate'         : ((1e-5, 0.5), float),
                    'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                    'momentum'              : (1e-3, 1e-2, 1e-1, 0.5, 0.9),
                    'momentum_type'         : 'normal',
                    'L1_lambda'             : None,
                    'L2_lambda'             : None,
                    'cost'                  : 'mse',
                    'stopping_criteria'     : DD({
                                                'max_epoch'         : 100,
                                                'epoch_look_back'   : 10,
                                                'cost'              : 'mse',
                                                'percent_decrease'  : 0.05
                                                }) # end stopping_criteria
                    }), # end learning_rule

            #===========================[ Dataset ]===========================#
            'dataset' : DD({
                    # 'type'                  : 'Laura_Warp_Blocks_500_Tanh',
                    # 'type'                 : 'Laura_Warp_Blocks_180_Tanh_Dropout',
                    # 'type'                  : 'Laura_Cut_Warp_Blocks_300',
                    # 'type'                  : 'Laura_Blocks_180_Tanh_Tanh',
                    # 'type'                  : 'Laura_Blocks_180_Tanh_Tanh_Dropout',
                    # 'type'                  : 'Laura_Blocks_500_Tanh_Sigmoid',
                    # 'type'                  : 'Laura_Blocks_500',
                    # 'type'                  : 'Laura_Blocks',
                    # 'type'                  : 'Laura_Warp_Blocks',
                    # 'type'                  : 'Laura_Warp_Standardize_Blocks',
                    # 'type'                  : 'Laura_Standardize_Blocks',

                    'type'                  : 'Laura_Scale_Warp_Blocks_500_Tanh',
                    # 'type'                  : 'Laura_Scale_Warp_Blocks_180_Tanh_Dropout',

                    # 'type'                  : 'Mnist',

                    'feature_size'          : 500,
                    'train_valid_test_ratio': [8, 1, 1],

                    'preprocessor'          : None,
                    # 'preprocessor'          : 'Scale',
                    # 'preprocessor'          : 'GCN',
                    # 'preprocessor'          : 'LogGCN',
                    # 'preprocessor'          : 'Standardize',

                    'batch_size'            : (50, 100, 150, 200),
                    'num_batches'           : None,
                    'iter_class'            : 'SequentialSubsetIterator',
                    'rng'                   : None
                    }), # end dataset

            #============================[ Layers ]===========================#
            'num_layers' : 1,

            'hidden1' : DD({
                    'name'                  : 'hidden1',
                    'type'                  : 'Tanh',
                    'dim'                   : 120,

                    'dropout_below'         : None,
                    # 'dropout_below'         : (0.3, 0.4, 0.5),
                    # 'dropout_below'         : 0.5,

                    # 'blackout_below'        : None,
                    'blackout_below'         : 0.5

                    }), # end hidden_layer

            'hidden2' : DD({
                    'name'                  : 'hidden2',
                    'type'                  : 'RELU',
                    'dim'                   : 100,
                    'dropout_below'         : None,

                    'blackout_below'        : None
                    }), # end hidden_layer

            'h2_mirror' : DD({
                    'name'                  : 'h2_mirror',
                    'type'                  : 'RELU',
                    # 'dim'                   : 2049, # dim = input.dim
                    'dropout_below'         : None,

                    'blackout_below'        : None
                    }), # end output_layer

            'h1_mirror' : DD({
                    'name'                  : 'h1_mirror',
                    'type'                  : 'Tanh',
                    # 'dim'                   : 2049, # dim = input.dim

                    'dropout_below'         : None,
                    # 'dropout_below'         : 0.5,

                    'blackout_below'        : None
                    }) # end output_layer

            }), # end autoencoder

    ########################[Laura_Continue]########################
    ##################################################################

    'Laura_Continue' : DD({
        'model' : DD({
                'rand_seed'             : 252
                }), # end mlp

        'log' : DD({

                'experiment_name'       : 'AE1003_Scale_Warp_Blocks_3Layers_finetune_2049_120_tanh_tanh_gpu_noisy',



                'description'           : '',
                'save_outputs'          : True,
                'save_learning_rule'      : True,
                'save_model'            : True,
                'save_to_database_name' : 'Laura2.db'
                }), # end log


        'learning_rule' : DD({
                'max_col_norm'          : 50,
                'learning_rate'         : 0.002109157240622,
                # 'learning_rate'         : ((1e-5, 9e-1), float),
                # 'learning_rate'         : 0.01,
                'momentum'              : 0.01,
                # 'momentum'              : 0.05,
                'momentum_type'         : 'normal',
                'L1_lambda'             : None,
                'L2_lambda'             : None,
                'cost'                  : 'mse',
                'stopping_criteria'     : DD({
                                            'max_epoch'         : 100,
                                            'epoch_look_back'   : 10,
                                            'cost'              : 'mse',
                                            'percent_decrease'  : 0.05
                                            }) # end stopping_criteria
                }), # end learning_rule

        #===========================[ Dataset ]===========================#
        'dataset' : DD({
                # 'type'                  : 'Laura_Warp_Blocks_500',
                # 'type'                  : 'Laura_Blocks_500',
                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',
                # 'type'                  : 'Mnist_Blocks',
                'feature_size'          : 2049,
                'train_valid_test_ratio': [8, 1, 1],

                # 'preprocessor'          : None,
                'preprocessor'          : 'Scale',
                # 'preprocessor'          : 'GCN',
                # 'preprocessor'          : 'LogGCN',
                # 'preprocessor'          : 'Standardize',

                'batch_size'            : 50,
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        # #============================[ Layers ]===========================#

        'hidden1' : DD({
                'name'                  : 'hidden1',
                'model'                 : 'AE1002_Scale_Warp_Blocks_3Layers_finetune_2049_120_tanh_tanh_gpu_noisy_20141003_0046_17962047',

                'dropout_below'         : None,
                }), # end hidden_layer


        }), # end autoencoder

    ########################[Laura_Two_Layers]########################
    ##################################################################

    'Laura_Two_Layers' : DD({
        'model' : DD({
                'rand_seed'             : None
                }), # end mlp

        'log' : DD({
                'experiment_name'       : 'AE1004_Scale_Warp_Blocks_2Layers_finetune_2049_120_tanh_tanh_gpu_clean',

                'description'           : '',
                'save_outputs'          : True,
                'save_learning_rule'      : True,
                'save_model'            : True,
                'save_to_database_name' : 'Laura2.db'
                }), # end log


        'learning_rule' : DD({
                # 'max_col_norm'          : (1, 10, 50),
                'max_col_norm'          : 50,
                'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                # 'learning_rate'         : ((1e-5, 9e-1), float),
                # 'learning_rate'         : 0.01,
                'momentum'              : (1e-3, 1e-2, 1e-1, 0.5, 0.9),
                # 'momentum'              : 0.05,
                'momentum_type'         : 'normal',
                'L1_lambda'             : None,
                'L2_lambda'             : None,
                'cost'                  : 'mse',
                'stopping_criteria'     : DD({
                                            'max_epoch'         : 100,
                                            'epoch_look_back'   : 10,
                                            'cost'              : 'mse',
                                            'percent_decrease'  : 0.05
                                            }) # end stopping_criteria
                }), # end learning_rule

        #===========================[ Dataset ]===========================#
        'dataset' : DD({

                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',

                'feature_size'          : 2049,
                'train_valid_test_ratio': [8, 1, 1],

                # 'preprocessor'          : None,
                'preprocessor'          : 'Scale',
                # 'preprocessor'          : 'GCN',
                # 'preprocessor'          : 'LogGCN',
                # 'preprocessor'          : 'Standardize',
                'batch_size'            : (50, 100, 150, 200),
                'num_batches'           : None,
                'iter_class'            : 'SequentialSubsetIterator',
                'rng'                   : None
                }), # end dataset

        # #============================[ Layers ]===========================#

        'hidden1' : DD({
                'name'                  : 'hidden1',

                # 'model'                 : 'AE0911_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140912_2337_04263067',
                'model'                 : 'AE0930_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140930_1345_29800576',
                'dropout_below'         : None,
                # 'dropout_below'         : (0.1, 0.2, 0.3, 0.4, 0.5),
                # 'dropout_below'         : 0.1,
                }), # end hidden_layer

        'hidden2' : DD({
                'name'                  : 'hidden2',

                # 'model'                 : 'AE1001_Warp_Blocks_500_120_tanh_tanh_gpu_clean_20141003_0113_02206401',
                'model'                 : 'AE1003_Scale_Warp_Blocks_500_120_tanh_tanh_gpu_clean_20141004_0640_00423949',
                'dropout_below'         : None,
                })
        }), # end autoencoder

    ########################[Laura_Three_Layers]########################
    ##################################################################

    'Laura_Three_Layers' : DD({
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

                'experiment_name'       : 'AE1002_Scale_Warp_Blocks_3Layers_finetune_2049_120_tanh_tanh_gpu_noisy',
                # 'experiment_name'       : 'AE1002_Scale_Warp_Blocks_3Layers_finetune_2049_120_tanh_tanh_gpu_clean',


                'description'           : '',
                'save_outputs'          : True,
                'save_learning_rule'      : True,
                'save_model'            : True,
                'save_to_database_name' : 'Laura2.db'
                }), # end log


        'learning_rule' : DD({
                'max_col_norm'          : (1, 10, 50),
                # 'max_col_norm'          : 50,
                # 'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                'learning_rate'         : ((1e-5, 9e-1), float),
                # 'learning_rate'         : 0.01,
                'momentum'              : (1e-3, 1e-2, 1e-1, 0.5, 0.9),
                # 'momentum'              : 0.05,
                'momentum_type'         : 'normal',
                'L1_lambda'             : None,
                'L2_lambda'             : None,
                'cost'                  : 'mse',
                'stopping_criteria'     : DD({
                                            'max_epoch'         : 100,
                                            'epoch_look_back'   : 10,
                                            'cost'              : 'mse',
                                            'percent_decrease'  : 0.05
                                            }) # end stopping_criteria
                }), # end learning_rule

        #===========================[ Dataset ]===========================#
        'dataset' : DD({

                # 'type'                  : 'Laura_Blocks',
                'type'                  : 'Laura_Warp_Blocks',

                'feature_size'          : 2049,
                'train_valid_test_ratio': [8, 1, 1],

                # 'preprocessor'          : None,
                'preprocessor'          : 'Scale',
                # 'preprocessor'          : 'GCN',
                # 'preprocessor'          : 'LogGCN',
                # 'preprocessor'          : 'Standardize',
                'batch_size'            : (50, 100, 150, 200),
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

                'model'                 : 'AE1002_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_dropout_20141001_0321_33382955',
                # 'model'                 : 'AE0930_Scale_Warp_Blocks_2049_500_tanh_tanh_gpu_clean_20140930_1345_29800576',

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

                'model'                 : 'AE1001_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_dropout_20141001_2158_16765065',
                # 'model'                 : 'AE1001_Scale_Warp_Blocks_500_180_tanh_tanh_gpu_clean_20141002_0348_53679208',

                'dropout_below'         : None,
                }), # end hidden_layer

        'hidden3' : DD({
                'name'                  : 'hidden3',
                # 'model'                 : 'AE0915_Warp_Blocks_180_120_tanh_gpu_dropout_clean_20140916_1028_26875210',
                # 'model'                 : 'AE0918_Warp_Blocks_180_120_tanh_tanh_gpu_dropout_20140919_1649_54631649',

                # 'model'                 : 'AE0914_Blocks_180_120_tanh_tanh_gpu_clean_20140918_0119_40376829',
                # 'model'                 : 'AE0919_Blocks_180_120_tanh_tanh_gpu_dropout_20140919_1345_22865393',

                # 'model'                 : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_dropout_20141002_1711_48207269',
                'model'                 : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_dropout_20141002_1457_08966968',
                # 'model'                 : 'AE1001_Scale_Warp_Blocks_180_120_tanh_tanh_gpu_clean_20141002_1713_16791523',

                'dropout_below'         : None,
                }), # end hidden_layer


        }), # end autoencoder

    #####################[Two_Layers_No_Transpose]######################
    ####################################################################

    'Laura_Two_Layers_No_Transpose' : DD({

        'model' : DD({
                'rand_seed'             : None
                }), # end mlp

        'log' : DD({
                'experiment_name'       : 'AE0730_No_Transpose_Warp_Blocks_180_64',
                'description'           : '',
                'save_outputs'          : True,
                'save_learning_rule'      : True,
                'save_model'            : True,
                'save_to_database_name' : 'Laura.db'
                }), # end log


        'learning_rule' : DD({
                'max_col_norm'          : (1, 10, 50),
                'learning_rate'         : (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5),
                # 'learning_rate'         : ((1e-5, 9e-1), float),
                # 'learning_rate'         : 0.01,
                'momentum'              : (1e-3, 1e-2, 1e-1, 0.5, 0.9),
                # 'momentum'              : 0.05,
                'momentum_type'         : 'normal',
                'L1_lambda'             : None,
                'L2_lambda'             : None,
                'cost'                  : 'mse',
                'stopping_criteria'     : DD({
                                            'max_epoch'         : 100,
                                            'epoch_look_back'   : 10,
                                            'cost'              : 'mse',
                                            'percent_decrease'  : 0.05
                                            }) # end stopping_criteria
                }), # end learning_rule

        #===========================[ Dataset ]===========================#
        'dataset' : DD({
                'type'                  : 'Laura_Warp_Blocks_180',
                # 'type'                  : 'Laura_Cut_Warp_Blocks_300',
                # 'type'                  : 'Laura_Blocks_500',
                # 'type'                  : 'Laura_Blocks',
                # 'type'                  : 'Laura_Warp_Blocks',
                'feature_size'          : 180,
                'train_valid_test_ratio': [8, 1, 1],
                'preprocessor'          : None,
                # 'preprocessor'          : 'Scale',
                # 'preprocessor'          : 'GCN',
                # 'preprocessor'          : 'LogGCN',
#                     'preprocessor'          : 'Standardize',
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
                'dim'                   : 64,
                'dropout_below'         : (0.1, 0.2, 0.3, 0.4, 0.5),
                # 'dropout_below'         : 0.1,
                }), # end hidden_layer


        'h1_mirror' : DD({
                'name'                  : 'h1_mirror',
                'type'                  : 'RELU',
                # 'dim'                   : 2049, # dim = input.dim
                'dropout_below'         : None
                }) # end output_layer


        }), # end autoencoder

    }) # end model_config
