import smartNN.datasets.iterator as iter


class IterMatrix(object):

    def __init__(self, X, y, iter_class, batch_size, num_batches=None, rng=None):
        self.X = X
        self.y = y
        self.iter_class = iter_class
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rng = rng
    
    def __iter__(self):
        return getattr(iter, self.iter_class)(self.dataset_size(), self.batch_size, 
                                                    self.num_batches, self.rng)
    
    def dataset_size(self):
        return self.X.shape[0]
    
    def feature_size(self):
        return self.X.shape[1]
    
    def target_size(self):
        return self.y.shape[1]
        

class Dataset(object):

    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test
        
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
   
        
    
        
    
   
        
    
    