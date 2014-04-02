from datetime import datetime
import os
import logging
import cPickle

class Log:

    def __init__(self, experiment_id, description=None, 
                save_outputs=False, save_hyperparams=False, save_model=False,
                send_to_database=False):
                
        dt = datetime.now()
        dt = dt.strftime('%Y%m%d_%H%M_%S%f')
        save_dir = os.environ['smartNN_SAVE_PATH']
        self.exp_dir = save_dir + '/' + experiment_id + '_' + dt
    
        os.mkdir(self.exp_dir)
    
        if save_outputs:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
    
            ch = logging.FileHandler(filename=self.exp_dir+'/outputs.log')
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            if description is not None:
                self.logger.info('Description: ' + description)
                
    def save_outputs(self, outputs):
        
        dt = datetime.now()
        dt = dt.strftime('%Y-%m-%d %H:%M')
        self.logger.info('Time: ' + dt)
        
        for (name, val) in outputs:
            self.logger.info(name + ': ' + str(val))
            
        self.logger.info('-------[ End of Epoch Logging ]-------\n')
                    
    def save_model(self, model):
        with open(self.exp_dir+'/model.pkl', 'wb') as pkl_file:
            cPickle.dump(model, pkl_file)
    
    def save_hyperparams(self, learning_rule):
        with open(self.exp_dir+'/hyperparams.pkl', 'wb') as pkl_file:
            cPickle.dump(learning_rule, pkl_file)      
    
    def send_to_database(self):
        pass
        
        
        
        
        