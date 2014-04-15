from datetime import datetime
import os
import logging
import cPickle
import sqlite3

class Log:

    def __init__(self, experiment_id, description=None, 
                save_outputs=False, save_hyperparams=False, save_model=False,
                send_to_database=None):
                
        self.experiment_id = experiment_id
        self.description = description
        self.save_outputs = save_outputs
        self.save_hyperparams = save_hyperparams
        self.save_model = save_model
        self.send_to_database = send_to_database
        
        dt = datetime.now()
        dt = dt.strftime('%Y%m%d_%H%M_%S%f')
        save_dir = os.environ['smartNN_SAVE_PATH'] + '/log'
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        self.exp_dir = save_dir + '/' + experiment_id + '_' + dt
        self.exp_dir_name = os.path.split(self.exp_dir)[-1]

        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
    
        if save_outputs:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
    
            ch = logging.FileHandler(filename=self.exp_dir+'/outputs.log')
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            self.logger.info('experiment_id: ' + self.experiment_id)
            
            if description is not None:
                self.logger.info('Description: ' + self.description)
        
        if send_to_database:
            self.first_time_record = True
                
    def _save_outputs(self, outputs):
        
        dt = datetime.now()
        dt = dt.strftime('%Y-%m-%d %H:%M')
        self.logger.info('Time: ' + dt)
        
        for (name, val) in outputs:
            self.logger.info(name + ': ' + str(val))
            
        self.logger.info('[ Outputs Logged to: %s ]\n' %self.exp_dir_name)
                    
    def _save_model(self, model):
        with open(self.exp_dir+'/model.pkl', 'wb') as pkl_file:
            cPickle.dump(model, pkl_file)
    
    def _save_hyperparams(self, learning_rule):
        with open(self.exp_dir+'/hyperparams.pkl', 'wb') as pkl_file:
            cPickle.dump(learning_rule, pkl_file)      
    
    def _send_to_database(self, epoch, dataset, learning_rule, train_error, 
                        valid_error, test_error, batch_size, 
                        num_layers, layers_struct, preprocessor):
                        
        conn = sqlite3.connect(os.environ['smartNN_DATABASE_PATH'] + 
                                '/' + self.send_to_database)
        cur = conn.cursor()
        result_cost_type = learning_rule.stopping_criteria['cost'].type
        
        cur.execute('CREATE TABLE IF NOT EXISTS ' + self.experiment_id + 
                    '(experiment_dir TEXT PRIMARY KEY NOT NULL,' +
                    'dataset TEXT,' +                     
                    'learning_rate REAL,' +
                    'max_col_norm INTEGER,' +
                    'momentum REAL,' + 
                    'momentum_type VARCHAR,' +
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

            cur.execute('INSERT INTO ' + self.experiment_id + 
                        ' VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);', 
                        [self.exp_dir_name,
                        dataset, 
                        learning_rule.learning_rate,
                        learning_rule.max_col_norm,
                        learning_rule.momentum,
                        learning_rule.momentum_type,
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
            
        else:
            cur.execute('UPDATE ' + self.experiment_id + ' SET ' + 
                        'dataset = ?,' +                     
                        'learning_rate = ?,' +
                        'max_col_norm = ?,' +
                        'momentum = ?,' + 
                        'momentum_type = ?,' +
                        'batch_size = ?,' + 
                        'num_layers = ?,' +
                        'layers_struct = ?,' +
                        'preprocessor = ?,' + 
                        'error_type = ?,' + 
                        'train_error = ?,' +
                        'valid_error = ?,' +
                        'test_error = ?,' +
                        'epoch = ? ' +
                        "WHERE experiment_dir='%s'"%self.exp_dir_name,
                        [dataset,
                        learning_rule.learning_rate,
                        learning_rule.max_col_norm,
                        learning_rule.momentum,
                        learning_rule.momentum_type,
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
        
        
                    
                    
        
        
        
        
        
        
        
        
        
        