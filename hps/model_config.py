from jobman import DD, flatten

model_config = DD({
        # One hidden layer Autoencoder
        
        #################################[ AE ]################################
        
        'AE' : DD({
        
            'mlp' : DD({
                    'rand_seed'             : ((1, 1000000), int)
                    }), # end mlp
            
            'log' : DD({
                    'experiment_id'         : 'AE23_LogGCN',
                    'description'           : 'first_layer_AE_Log_GCN_with_Sigmoid_internal_units',
                    'save_outputs'          : True,
                    'save_hyperparams'      : True,
                    'save_model'            : True,
                    'send_to_database'      : 'Database_Name.db'
                    }), # end log
            
            'learning_rule' : DD({
                    'max_col_norm'          : 10,
                    'learning_rate'         : ((0.0001, 0.9), float),
                    'momentum'              : ((0.0001, 0.9), float),
                    'momentum_type'         : 'normal',
                    'weight_decay'          : 0,
                    'cost'                  : 'entropy',
                    'stopping_criteria'     : DD({
                                                'max_epoch'         : 100,
                                                'epoch_look_back'   : 20,
                                                'cost'              : 'entropy',
                                                'percent_decrease'  : 0.005
                                                }) # end stopping_criteria
                    }), # end learning_rule
                    
            #===========================[ Dataset ]===========================#            
#             'dataset' : DD({
#                     'type'                  : 'Mnist',
#                     'train_valid_test_ratio': [8, 1, 1],
#                     'preprocessor'          : 'Scale',
# #                     'preprocessor'          : 'Standardize',
#                     'batch_size'            : 100,
#                     'num_batches'           : None,
#                     'iter_class'            : 'SequentialSubsetIterator',
#                     'rng'                   : None
#                     }), # end dataset
                        
            'dataset' : DD({
                    'type'                  : 'P276',
#                     'type'                  : 'P276_Scale_AE_output',
                    'train_valid_test_ratio': [8, 1, 1],
#                     'preprocessor'          : 'Scale',
#                     'preprocessor'          : 'GCN',
                    'preprocessor'          : 'LogGCN',
#                     'preprocessor'          : 'Standardize',
                    'batch_size'            : 100,
                    'num_batches'           : None,
                    'iter_class'            : 'SequentialSubsetIterator',
                    'rng'                   : None
                    }), # end dataset
                    
            #============================[ Layers ]===========================#               
            'hidden_layer' : DD({
                    'name'                  : 'hidden_layer',
                    'type'                  : 'Sigmoid',
                    'dim'                   : 500,
#                     'dim'                   : 64,
                    'dropout_below'         : None
                    }), # end hidden_layer
            
            'output_layer' : DD({
                    'name'                  : 'output_layer',
                    'type'                  : 'Sigmoid',
#                      dim not needed as output dim = input dim
#                     'dim'                   : None,
                    'dropout_below'         : None
                    }) # end output_layer
            }), # end autoencoder
            
        ############################[ AE_Second_Stack ]############################

        'AE_Second_Stack' : DD({
            
            'mlp' : DD({
                    'rand_seed'             : ((1, 1000000), int)
                    }), # end mlp
            
            'log' : DD({
                    'experiment_id'         : 'AE22_layer2_LogGCN',
                    'description'           : 'second_layer_AE_Log_GCN_with_Sigmoid_internal_units',
                    'save_outputs'          : True,
                    'save_hyperparams'      : True,
                    'save_model'            : True,
                    'send_to_database'      : 'Database_Name.db'
                    }), # end log
            
            'learning_rule' : DD({
                    'max_col_norm'          : 10,
                    'learning_rate'         : ((0.0001, 0.9), float),
                    'momentum'              : ((0.0001, 0.9), float),
                    'momentum_type'         : 'normal',
                    'weight_decay'          : 0,
                    'cost'                  : 'entropy',
                    'stopping_criteria'     : DD({
                                                'max_epoch'         : 100,
                                                'epoch_look_back'   : 20,
                                                'cost'              : 'entropy',
                                                'percent_decrease'  : 0.005
                                                }) # end stopping_criteria
                    }), # end learning_rule
                    
            #===========================[ Dataset ]===========================#            
#             'dataset' : DD({
#                     'type'                  : 'Mnist',
#                     'train_valid_test_ratio': [8, 1, 1],
#                     'preprocessor'          : 'Scale',
# #                     'preprocessor'          : 'Standardize',
#                     'batch_size'            : 100,
#                     'num_batches'           : None,
#                     'iter_class'            : 'SequentialSubsetIterator',
#                     'rng'                   : None
#                     }), # end dataset
                        
            'dataset' : DD({
                    'type'                  : 'P276',
#                     'type'                  : 'P276_Scale_AE_output',
                    'train_valid_test_ratio': [8, 1, 1],
#                     'preprocessor'          : 'Scale',
#                     'preprocessor'          : 'GCN',
                    'preprocessor'          : 'LogGCN',
#                     'preprocessor'          : 'Standardize',
                    'batch_size'            : 100,
                    'num_batches'           : None,
                    'iter_class'            : 'SequentialSubsetIterator',
                    'rng'                   : None
                    }), # end dataset
                    
            #============================[ Layers ]===========================#               
            'hidden_layer' : DD({
                    'name'                  : 'hidden_layer',
                    'type'                  : 'Sigmoid',
#                     'dim'                   : 500,
                    'dim'                   : 64,
                    'dropout_below'         : None
                    }), # end hidden_layer
            
            'output_layer' : DD({
                    'name'                  : 'output_layer',
                    'type'                  : 'Sigmoid',
#                      dim not needed as output dim = input dim
#                     'dim'                   : None,
                    'dropout_below'         : None
                    }) # end output_layer
            }), # end autoencoder
            
        ###########################[ AE_Two_Layers ]###########################
            
            
                    # One hidden layer Autoencoder
        'AE_Two_Layers' : DD({
        
            'mlp' : DD({
#                     'rand_seed'             : ((123, 1000000), int)
                    }), # end mlp
            
            'log' : DD({
                    'experiment_id'         : 'AE15Double_GCN',
                    'description'           : 'Two_layers_AE_GCN_with_Sigmoid_internal_units',
                    'save_outputs'          : True,
                    'save_hyperparams'      : True,
                    'save_model'            : True,
                    'send_to_database'      : 'Database_Name.db'
                    }), # end log
            
            'learning_rule' : DD({
                    'max_col_norm'          : ((1, 10), int),
                    'learning_rate'         : ((0.001, 0.1), float),
                    'momentum'              : ((0.001, 0.1), float),
                    'momentum_type'         : 'normal',
                    'weight_decay'          : 0,
                    'cost'                  : 'entropy',
                    'stopping_criteria'     : DD({
                                                'max_epoch'         : 100,
                                                'epoch_look_back'   : 10,
                                                'cost'              : 'entropy',
                                                'percent_decrease'  : 0.005
                                                }) # end stopping_criteria
                    }), # end learning_rule
                    
            #===========================[ Dataset ]===========================#            
#             'dataset' : DD({
#                     'type'                  : 'Mnist',
#                     'train_valid_test_ratio': [8, 1, 1],
#                     'preprocessor'          : 'Scale',
# #                     'preprocessor'          : 'Standardize',
#                     'batch_size'            : 100,
#                     'num_batches'           : None,
#                     'iter_class'            : 'SequentialSubsetIterator',
#                     'rng'                   : None
#                     }), # end dataset
                        
            'dataset' : DD({
                    'type'                  : 'P276',
#                     'type'                  : 'P276_Scale_AE_output',
                    'train_valid_test_ratio': [8, 1, 1],
#                     'preprocessor'          : 'Scale',
                    'preprocessor'          : 'GCN',
#                     'preprocessor'          : 'Standardize',
                    'batch_size'            : ((32, 200), int),
                    'num_batches'           : None,
                    'iter_class'            : 'SequentialSubsetIterator',
                    'rng'                   : None
                    }), # end dataset
                    
            #============================[ Layers ]===========================#               
            'hidden1' : DD({
                    'model_name'            : 'AE15_GCN_20140414_2342_39424209',
#                     'model_name'            : 'AE15_Scale_20140414_2349_19835883'

                    }), # end hidden_layer
            
            'hidden2' : DD({
                    'model_name'            : 'AE15_2_GCN_20140415_0756_44509622',
#                     'model_name'            : 'AE15_2_Scale_20140415_0804_48200863'
                    }) # end output_layer
            }) # end autoencoder
            
    }) # end model_config
    