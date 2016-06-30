from __future__ import absolute_import
from __future__ import print_function

import matplotlib
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano.compile.ops import as_op
from mozi.utils.progbar import Progbar
from mozi.utils.train_object_utils import is_shared_var

import tarfile, inspect, os
from six.moves.urllib.request import urlretrieve

floatX = theano.config.floatX

def duplicate_param(name, tensor_list):

    for param in tensor_list:
        if param.name is name:
            return True

    return False


def tile_raster_graphs(dct_reconstruct, orig, ae_reconstruct, tile_shape, tile_spacing=(0.1,0.1),
                        slice=(0,-1), axis=None, legend=True):
    """
    DESCRIPTION:
        compare the original and the reconstructed examples by plot them on the same graph
    PARAM:
        orig / ae_reconstruct / dct_reconstruct : 2d numpy array of axis label [example, feature]
        tile_shape : tuple
        tile_spacing : tuple
        slice : index [start:end]
            gives the range of values in the example to plot
        axis : list [x_min, x_max, y_min, y_max]
            sets the bounds of the x and y axis
    RETURN:
        matplotlib.plot object
    """

    assert orig.shape == ae_reconstruct.shape, 'orig ' + str(orig.shape) + ' and reconstruct ' + \
        str(ae_reconstruct.shape) + ' shapes are different'

    # make a little extra space between the subplots
    plt.subplots_adjust(wspace=tile_spacing[0], hspace=tile_spacing[1])

    num_examples = orig.shape[0]
    if num_examples > tile_shape[0] * tile_shape[1]:
        num_examples = tile_shape[0] * tile_shape[1]

    for i in xrange(0, num_examples):
        plt.subplot(tile_shape[0], tile_shape[1], i+1)
        plt.plot(orig[i][slice[0]:slice[1]], 'b-', label='orig')
        plt.plot(ae_reconstruct[i][slice[0]:slice[1]], 'g-', label='AE reconstruct')
        plt.plot(dct_reconstruct[i][slice[0]:slice[1]], 'r-', label='DCT reconstruct')
        if legend:
            plt.legend(loc='best')
        if axis is None:
            plt.axis('tight')
        else:
            plt.axis(axis)
    return plt

def make_one_hot(X, onehot_size):
    """
    DESCRIPTION:
        Make a one-hot version of X
    PARAM:
        X: 1d numpy with each value in X representing the class of X
        onehot_size: length of the one hot vector
    RETURN:
        2d numpy tensor, with each row been the onehot vector
    """

    rX = np.zeros((len(X), onehot_size), dtype=theano.config.floatX)
    for i in xrange(len(X)):
        rX[i, X[i]] = 1

    return rX

def get_file(fpath, origin, untar=False):
    datadir = os.path.dirname(fpath)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if not os.path.exists(fpath):
        print('Downloading data from',  origin)

        global progbar
        progbar = None
        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count*block_size)

        urlretrieve(origin, fpath, dl_progress)
        progbar = None

    dirname = ""
    if untar:
        tfile = tarfile.open(fpath, 'r:*')
        names = tfile.getnames()
        dirname = names[0]
        not_exists = [int(not os.path.exists("{}/{}".format(datadir, fname))) for fname in names]
        if sum(not_exists) > 0:
            print('Untaring file...')
            tfile.extractall(path=datadir)
        else:
            print('Files already downloaded and untarred')
        tfile.close()

    return "{}/{}".format(datadir, dirname)


@as_op(itypes=[theano.tensor.fmatrix],
       otypes=[theano.tensor.fmatrix])
def theano_unique(a):
    return numpy.unique(a)


def get_from_module(identifier, module_params, module_name, instantiate=False):
    if type(identifier) is str:
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' + str(identifier))
        if instantiate:
            return res()
        else:
            return res
    return identifier


def make_tuple(*args):
    return args


def gpu_to_cpu_model(model):
    for layer in model.layers:
        for member, value in layer.__dict__.items():
            if is_shared_var(value):
                layer.__dict__[member] = T._shared(np.array(value.get_value(), floatX),
                                          name=value.name, borrow=False)
        for i in xrange(len(layer.params)):
            if is_shared_var(layer.params[i]):
                layer.params[i] = T._shared(np.array(layer.params[i].get_value(), floatX),
                                          name=layer.params[i].name, borrow=False)
    return model


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
        Pad each sequence to the same length:
        the length of the longuest sequence.

        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.

        Supports post-padding and pre-padding (default).

    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x
