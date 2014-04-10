import logging
logger = logging.getLogger(__name__)
import smartNN.datasets.iterator as iter

class IterMatrix(object):

    def __init__(self, X, y, preprocessor=None, iter_class='SequentialSubsetIterator', 
                batch_size=100, num_batches=None, rng=None):

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

    def __init__(self, train, valid, test,
                preprocessor=None, iter_class='SequentialSubsetIterator', 
                batch_size=100, num_batches=None, rng=None):
    
        ''' 
        DESCRIPTION: Interface that contains three IterMatrix
        PARAM:
            train: list
            valid: list
            test: list    
        '''
                
        self.preprocessor = preprocessor
        self.iter_class = iter_class
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = rng
        
        if train is None:
            logger.warning('Train set is empty!')
            self.train = None
        else:
            self.train = IterMatrix(train[0], train[1], preprocessor=self.preprocessor, 
                                    iter_class=self.iter_class, batch_size=self.batch_size, 
                                    num_batches=self.num_batches, rng=self.rng)
        
        if valid is None:
            logger.warning('Valid set is empty! It is needed for stopping of training')
            self.valid = None
        else:
            self.valid = IterMatrix(valid[0], valid[1], preprocessor=self.preprocessor, 
                                    iter_class=self.iter_class, batch_size=self.batch_size, 
                                    num_batches=self.num_batches, rng=self.rng)
        
        if test is None:
            logger.warning('Test set is empty! It is needed for saving the best model')
            self.test = None
        else:
            self.test = IterMatrix(test[0], test[1], preprocessor=self.preprocessor, 
                                    iter_class=self.iter_class, batch_size=self.batch_size, 
                                    num_batches=self.num_batches, rng=self.rng)
        
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
   
        
    
        
    
   
        
    
    