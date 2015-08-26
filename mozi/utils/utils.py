from __future__ import absolute_import
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import theano
import numpy as np
import matplotlib.pyplot as plt
from theano.compile.ops import as_op
from mozi.utils.progbar import Progbar

import tarfile, inspect, os
from six.moves.urllib.request import urlretrieve

def split_list(tuple_list):
    """
    DESCRIPTION:
        split a list of tuples into two lists whereby one list contains the first elements
        of the tuples and the other list contains the second elements.
    PARAM:
        tuple_list: a list of tuples, example tuple_list = [('a', 1), ('b', 2)]
    RETURN:
        two lists, example from above tuple_list will be split into ['a', 'b'] and [1, 2]
    """
    ls_A = []
    ls_B = []

    for tuple in tuple_list:
        ls_A.append(tuple[0])
        ls_B.append(tuple[1])

    return ls_A, ls_B

def generate_shared_list(ls):
    """
    DESCRIPTION:
        generate a list of shared variables that matched the length of ls
    PARAM:
        ls: the list used for generating the shared variables
    RETURN:
        a list of shared variables initialized to 0 of len(ls)
    """
    rlist = []

    for i in xrange(len(ls)):
        rlist.append(theano.shared(np.array(0., dtype=theano.config.floatX)))

    return rlist

def merge_lists(ls_A, ls_B):
    """
    DESCRIPTION:
        merge two lists of equal length into into a list of tuples
    PARAM:
        ls_A: first list
        ls_B: second list
    RETURN:
        a list of tuples
    """

    assert len(ls_A) == len(ls_B), 'two lists of different length'

    rlist = []
    for a, b in zip(ls_A, ls_B):
        rlist.append((a,b))

    return rlist

def get_shared_values(shared_ls):
    """
    DESCRIPTION:
        get a list of values from a list of shared variables
    PARAM:
        shared_ls: list of shared variables
    RETURN:
        numpy array of the list of values
    """

    val_ls = []
    for var in shared_ls:
        val_ls.append(var.get_value())

    return np.asarray(val_ls, dtype=theano.config.floatX)


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


def is_shared_var(var):
    return var.__class__.__name__ == 'TensorSharedVariable' or \
            var.__class__.__name__ == 'CudaNdarraySharedVariable'
