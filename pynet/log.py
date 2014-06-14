from datetime import datetime
import os
import sys
import logging
import cPickle
import sqlite3

class Log:

    def __init__(self, experiment_name="experiment", description=None, 
                save_outputs=False, save_hyperparams=False, save_model=False,
                send_to_database=None, logger=None):
                
        self.experiment_name = experiment_name
        self.description = description
        self.save_outputs = save_outputs
        self.save_hyperparams = save_hyperparams
        self.save_model = save_model
        self.send_to_database = send_to_database
        
        dt = datetime.now()
        dt = dt.strftime('%Y%m%d_%H%M_%S%f')
        
        self.exp_id = experiment_name + '_' + dt

        if save_outputs or save_hyperparams or save_model:
            save_dir = os.environ['PYNET_SAVE_PATH'] + '/log'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        
            self.exp_dir = save_dir + '/' + self.exp_id
            if not os.path.exists(self.exp_dir):
                os.mkdir(self.exp_dir)
    
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        
        if save_outputs:
            ch = logging.FileHandler(filename=self.exp_dir+'/outputs.log')
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.logger.info('exp_id: ' + self.exp_id)
        
        if description is not None:
            self.logger.info('Description: ' + self.description)
        
        if send_to_database:
            self.first_time_record = True
                
    def _log_outputs(self, outputs):
        
        dt = datetime.now()
        dt = dt.strftime('%Y-%m-%d %H:%M')
        self.logger.info('Time: ' + dt)
        
        for (name, val) in outputs:
            self.logger.info(name + ': ' + str(val))
        
        if self.save_outputs:
            self.logger.info('[ outputs saved to: %s ]\n' %self.exp_id)
            
        print('\n')
                    
    def _save_model(self, model):
        with open(self.exp_dir+'/model.pkl', 'wb') as pkl_file:
            cPickle.dump(model, pkl_file)
    
    def _save_hyperparams(self, learning_rule):
        with open(self.exp_dir+'/hyperparams.pkl', 'wb') as pkl_file:
            cPickle.dump(learning_rule, pkl_file)      
    
    def _send_to_database(self, epoch, dataset, rand_seed, layers_dropout_below, learning_rule, train_error, 
                        valid_error, test_error, batch_size, 
                        num_layers, layers_struct, preprocessor):
                        
        conn = sqlite3.connect(os.environ['PYNET_DATABASE_PATH'] + 
                                '/' + self.send_to_database)
        cur = conn.cursor()
        result_cost_type = learning_rule.stopping_criteria['cost'].type
        
        cur.execute('CREATE TABLE IF NOT EXISTS ' + self.experiment_name + 
                    '(exp_id TEXT PRIMARY KEY NOT NULL,' +
                    'dataset TEXT,' +  
                    'weight_initialization_seed INT,' +
                    'layers_dropout_below TEXT,' +                   
                    'learning_rate REAL,' +
                    'max_col_norm INTEGER,' +
                    'momentum REAL,' + 
                    'momentum_type VARCHAR,' +
                    'L1_lambda REAL,' +
                    'L2_lambda REAL,' +
                    'batch_size INTEGER,' + 
                    'num_layers INTEGER,' +
                    'layers_struct TEXT,' + 
                    'preprocessor TEXT,' +
                    'error_type TEXT,' + 
                    'train_error REAL,' +
                    'valid_error REAL,' +
                    'test_error REAL,' +
                    'epoch);')


        
        if self.first_time_record:
            try:
                cur.execute('INSERT INTO ' + self.experiment_name + 
                            ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);', 
                            [self.exp_id,
                            dataset,
                            rand_seed,
                            layers_dropout_below,
                            learning_rule.learning_rate,
                            learning_rule.max_col_norm,
                            learning_rule.momentum,
                            learning_rule.momentum_type,
                            learning_rule.L1_lambda,
                            learning_rule.L2_lambda,
                            batch_size,
                            num_layers,
                            layers_struct,
                            preprocessor,
                            result_cost_type,
                            train_error,
                            valid_error,
                            test_error,
                            epoch])
                self.first_time_record = False
                
            except sqlite3.OperationalError as err:
                self.logger.error('sqlite3.OperationalError: ' + err.message)
                self.logger.error('Solution: Change the experiment_name in Log() to a new name, '
                        + 'or drop the same table name from the database. '
                        + 'experiment_name is used as the table name.')
                raise
            
        else:
            cur.execute('UPDATE ' + self.experiment_name + ' SET ' + 
                        'dataset = ?,' +  
                        'weight_initialization_seed = ?,' +
                        'layers_dropout_below = ?,' +                   
                        'learning_rate = ?,' +
                        'max_col_norm = ?,' +
                        'momentum = ?,' + 
                        'momentum_type = ?,' +
                        'L1_lambda = ?,' +
                        'L2_lambda = ?,' +
                        'batch_size = ?,' + 
                        'num_layers = ?,' +
                        'layers_struct = ?,' +
                        'preprocessor = ?,' + 
                        'error_type = ?,' + 
                        'train_error = ?,' +
                        'valid_error = ?,' +
                        'test_error = ?,' +
                        'epoch = ? ' +
                        "WHERE exp_id='%s'"%self.exp_id,
                        [dataset,
                        rand_seed,
                        layers_dropout_below,
                        learning_rule.learning_rate,
                        learning_rule.max_col_norm,
                        learning_rule.momentum,
                        learning_rule.momentum_type,
                        learning_rule.L1_lambda,
                        learning_rule.L2_lambda,
                        batch_size,
                        num_layers,
                        layers_struct,
                        preprocessor,
                        result_cost_type,
                        train_error,
                        valid_error,
                        test_error,
                        epoch
                        ])
        
        conn.commit()        
        conn.close()
        
        
                    
                    
        
        
        
        
        
        
        
        
        
        