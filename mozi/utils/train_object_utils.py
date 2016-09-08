from __future__ import absolute_import
from __future__ import print_function

import matplotlib
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano.compile.ops import as_op
from mozi.utils.progbar import Progbar

import tarfile, inspect, os
from six.moves.urllib.request import urlretrieve

floatX = theano.config.floatX

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


def is_shared_var(var):
    return var.__class__.__name__ == 'TensorSharedVariable' or \
            var.__class__.__name__ == 'CudaNdarraySharedVariable'


def merge_var(*vars):
    def absortvar(v):
        rvar = []
        if isinstance(v, (list, tuple)):
            rvar += v
        else:
            rvar.append(v)
        return rvar

    rvars = []
    for var in vars:
        rvars += absortvar(var)
    return rvars
