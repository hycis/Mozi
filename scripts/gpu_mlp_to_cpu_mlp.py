
import argparse
import cPickle
import theano.tensor as T
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import numpy as np
import copy
import theano
from pynet.model import MLP
import pynet.layer as layers

floatX = theano.config.floatX

parser = argparse.ArgumentParser(description='''Convert gpu pickle pynet model to cpu pickle pynet model''')
parser.add_argument('--gpu_model', metavar='Path', required=True, help='the path to the gpu model pickle file')
parser.add_argument('--cpu_model', metavar='Path', required=True, help='''path to save the cpu model pickle file''')
args = parser.parse_args()

print ('loading gpu mlp..')
fin = open(args.gpu_model)
gpu_model = cPickle.load(fin)

mlp = MLP(input_dim=gpu_model.input_dim)
for layer in gpu_model.layers:
    layerW = T._shared(np.array(layer.W.get_value(), floatX),
                        name=layer.W.name, borrow=False)
    layerb = T._shared(np.array(layer.b.get_value(), floatX),
                        name=layer.b.name, borrow=False)
    mlp_layer = getattr(layers, layer.__class__.__name__)(dim=layer.dim, name=layer.name,
                                                            W=layerW, b=layerb)
    mlp.add_layer(mlp_layer)
    print 'mlp layer', mlp_layer.name, mlp_layer.dim
print 'layers', mlp.layers

fout = open(args.cpu_model, 'wb')
cPickle.dump(mlp, fout)
print ('Done!')
fin.close()
fout.close()
