
from mozi.layers.activation import RELU, Softmax
from mozi.layers.normalization import LRN
from mozi.layers.convolution import Convolution2D, Pooling2D
from mozi.layers.linear import Linear
from mozi.layers.noise import Dropout
from mozi.layers.misc import Flatten
from mozi.layers.template import Template


class Alexnet(Template):

    def __init__(self, input_shape, output_dim):
        '''
        FIELDS:
            self.params: any params from the layer that needs to be updated
                         by backpropagation can be put inside self.params
        PARAMS:
            input_shape: tuple
                         shape of the input image with format (channel, height, width)
            output_dim: int
                        the output dimension of the model
        '''
        assert len(input_shape) == 3, 'input_shape must be a tuple or list of dim (channel, height, width)'
        c, h, w = input_shape

        valid = lambda x, y, kernel, stride : ((x-kernel)/stride + 1, (y-kernel)/stride + 1)
        full = lambda x, y, kernel, stride : ((x+kernel)/stride - 1, (y+kernel)/stride - 1)

        self.layers = []
        self.layers.append(Convolution2D(input_channels=3, filters=96, kernel_size=(11,11),
                                         stride=(4,4), border_mode='valid'))
        nh, nw = valid(h, w, 11, 4)
        self.layers.append(RELU())
        self.layers.append(LRN())
        self.layers.append(Pooling2D(poolsize=(3,3), stride=(2,2), mode='max'))
        nh, nw = valid(nh, nw, 3, 2)
        self.layers.append(Convolution2D(input_channels=96, filters=256, kernel_size=(5,5),
                                         stride=(1,1), border_mode='full'))
        nh, nw = full(nh, nw, 5, 1)
        self.layers.append(RELU())
        self.layers.append(LRN())
        self.layers.append(Pooling2D(poolsize=(3,3), stride=(2,2), mode='max'))
        nh, nw = valid(nh, nw, 3, 2)
        self.layers.append(Convolution2D(input_channels=256, filters=384, kernel_size=(3,3),
                                         stride=(1,1), border_mode='full'))
        nh, nw = full(nh, nw, 3, 1)
        self.layers.append(RELU())
        self.layers.append(Convolution2D(input_channels=384, filters=384, kernel_size=(3,3),
                                         stride=(1,1), border_mode='full'))
        nh, nw = full(nh, nw, 3, 1)
        self.layers.append(RELU())
        self.layers.append(Convolution2D(input_channels=384, filters=256, kernel_size=(3,3),
                                         stride=(1,1), border_mode='full'))
        nh, nw = full(nh, nw, 3, 1)
        self.layers.append(RELU())
        self.layers.append(Pooling2D(poolsize=(3,3), stride=(2,2), mode='max'))
        nh, nw = valid(nh, nw, 3, 2)

        self.layers.append(Flatten())
        self.layers.append(Linear(256*nh*nw,4096))
        self.layers.append(RELU())
        self.layers.append(Dropout(0.5))
        self.layers.append(Linear(4096,4096))
        self.layers.append(RELU())
        self.layers.append(Dropout(0.5))
        self.layers.append(Linear(4096,output_dim))
        self.layers.append(Softmax())

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def _test_fprop(self, state_below):
        for layer in self.layers:
            state_below = layer._test_fprop(state_below)
        return state_below

    def _train_fprop(self, state_below):
        for layer in self.layers:
            state_below = layer._train_fprop(state_below)
        return state_below
