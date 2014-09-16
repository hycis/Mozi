
import argparse
import cPickle
import theano.tensor as T
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
import numpy as np
import copy
import theano
from pynet.model import AutoEncoder
import pynet.layer as layers

floatX = theano.config.floatX

parser = argparse.ArgumentParser(description='''Convert gpu pickle pynet model to cpu pickle pynet model''')
parser.add_argument('--gpu_model', metavar='Path', required=True, help='the path to the gpu model pickle file')
parser.add_argument('--cpu_model', metavar='Path', required=True, help='''path to save the cpu model pickle file''')
args = parser.parse_args()

print ('loading gpu autoencoder..')
fin = open(args.gpu_model)
gpu_model = cPickle.load(fin)

ae = AutoEncoder(input_dim=gpu_model.input_dim)
for layer in gpu_model.encode_layers:
    layerW = T._shared(np.array(layer.W.get_value(), floatX),
                        name=layer.W.name, borrow=False)
    layerb = T._shared(np.array(layer.b.get_value(), floatX),
                        name=layer.b.name, borrow=False)
    encode_layer = getattr(layers, layer.__class__.__name__)(dim=layer.dim, name=layer.name,
                                                            W=layerW, b=layerb)
    ae.add_encode_layer(encode_layer)
    print 'encode layer', encode_layer.name, encode_layer.dim
print 'encode layers', ae.encode_layers

for ae_layer, gpu_layer in zip(reversed(ae.encode_layers), gpu_model.decode_layers):
    gpu_decode_layer_b = T._shared(np.array(gpu_layer.b.get_value(), floatX),
                        name=gpu_layer.b.name, borrow=False)
    decode_layer = getattr(layers, gpu_layer.__class__.__name__)(name=gpu_layer.name, dim=gpu_layer.dim,
                                                                W=ae_layer.W.T, b=gpu_decode_layer_b)
    ae.add_decode_layer(decode_layer)
    print 'decode layer', decode_layer.name, decode_layer.dim
print 'decode layers', ae.decode_layers
print 'layers', ae.layers
fout = open(args.cpu_model, 'wb')
cPickle.dump(ae, fout)
print ('Done!')
fin.close()
fout.close()
