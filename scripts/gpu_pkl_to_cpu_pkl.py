
import argparse
import cPickle
import theano.tensor as T
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import numpy as np
import copy
import theano
floatX = theano.config.floatX

parser = argparse.ArgumentParser(description='''Convert gpu pickle pynet model to cpu pickle pynet model''')
parser.add_argument('--gpu_model', metavar='Path', required=True, help='the path to the gpu model pickle file')
parser.add_argument('--cpu_model', metavar='Path', required=True, help='''path to save the cpu model pickle file''')
args = parser.parse_args()

print ('loading gpu model..')
fin = open(args.gpu_model)
gpu_model = cPickle.load(fin)

for layer in gpu_model.layers:
    if isinstance(layer.W,  CudaNdarraySharedVariable):
        print ('W is cuda, converting')
        layer.W = T._shared(np.array(layer.W.get_value(), floatX),
                            name='W_'+layer.name, borrow=True)
    if isinstance(layer.b,  CudaNdarraySharedVariable):
        print ('b is cuda, converting')
        layer.b = T._shared(np.array(layer.b.get_value(), floatX),
                            name='b_'+layer.name, borrow=True)

fout = open(args.cpu_model, 'wb')
cpu_model = cPickle.dump(gpu_model, fout)
print ('Done!')
fin.close()
fout.close()
