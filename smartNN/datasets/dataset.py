import logging
logger = logging.getLogger(__name__)
import smartNN.datasets.iterator as iter

class IterMatrix(object):

    def __init__(self, X, y, batch_size, num_batches, iter_class, preprocessor=None, rng=None):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.iter_class = iter_class
        self.rng = rng
        
        self.iterator = getattr(iter, self.iter_class)
        self.preprocessor = preprocessor
        
        if preprocessor is not None:
            logger.info('..applying preprocessing: ' + self.preprocessor.__class__.__name__)
            self.X = self.preprocessor.apply(self.X)
    
    def __iter__(self):
        return self.iterator(dataset_size=self.dataset_size(), 
                            batch_size=self.batch_size, 
                            num_batches=self.num_batches, 
                            rng=self.rng)
    
    def set_iterator(self, iterator):
        self.iterator = iterator
        
    def apply_preprocessor(self, preprocessor):
        self.X = preprocessor.apply(self.X)
    
    def dataset_size(self):
        return self.X.shape[0]
    
    def feature_size(self):
        return self.X.shape[1]
    
    def target_size(self):
        return self.y.shape[1]
        

class Dataset(object):

    def __init__(self, train, valid, test):
    
        ''' 
        DESCRIPTION: Interface that contains three IterMatrix
        PARAM:
            train: IterMatrix
            valid: IterMatrix
            test: IterMatrix    
        '''
                
        self.train = train
        self.valid = valid
        self.test = test
        
        if self.train is None:
            logger.warning('Train set is empty!')
        
        if self.valid is None:
            logger.warning('Valid set is empty! It is needed for stopping of training')
        
        if self.test is None:
            logger.warning('Test set is empty! It is needed for saving the best model')
        
        
    def get_train(self):
        return self.train
        
    def get_valid(self):
        return self.valid
        
    def get_test(self):
        return self.test
            
    def set_train(self, X, y):
        self.train.X = X
        self.train.y = y
            
    def set_valid(self, X, y):
        self.valid.X = X
        self.valid.y = y
            
    def set_test(self, X, y):
        self.test.X = X
        self.test.y = y
    
    def feature_size(self):
        return self.train.X.shape[1]
    
    def target_size(self):
        return self.train.y.shape[1]
   
        
    
        
    
   
        
    
    