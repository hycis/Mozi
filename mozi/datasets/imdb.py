
import logging
logger = logging.getLogger(__name__)
import os
import cPickle
import numpy as np
import theano
floatX = theano.config.floatX

from mozi.utils.utils import get_file, make_one_hot, pad_sequences
from mozi.datasets.dataset import SingleBlock

class IMDB(SingleBlock):

    def __init__(self, nb_words=None, skip_top=0, maxlen=None, seed=113,
                 pad_zero=False, start_char=1, oov_char=2, index_from=3, **kwargs):
        '''
        adapted from keras
        '''
        im_dir = os.environ['MOZI_DATA_PATH'] + '/imdb/'
        path = "https://s3.amazonaws.com/text-datasets/imdb.pkl"
        im_dir = get_file(fpath="{}/imdb.pkl".format(im_dir), origin=path, untar=False)
        with open('{}/imdb.pkl'.format(im_dir)) as fin:
            X, labels = np.load(fin)
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(labels)

        if start_char is not None:
            X = [[start_char] + [w + index_from for w in x] for x in X]
        elif index_from:
            X = [[w + index_from for w in x] for x in X]

        if maxlen:
            new_X = []
            new_labels = []
            for x, y in zip(X, labels):
                if len(x) < maxlen:
                    new_X.append(x)
                    new_labels.append(y)
            X = new_X
            labels = new_labels

        if not nb_words:
            nb_words = max([max(x) for x in X])

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
        else:
            nX = []
            for x in X:
                nx = []
                for w in x:
                    if (w >= nb_words or w < skip_top):
                        nx.append(w)
                nX.append(nx)
            X = nX

        if pad_zero and maxlen:
            X = pad_sequences(X, maxlen=maxlen)
        super(IMDB, self).__init__(X=np.asarray(X), y=np.asarray(labels).reshape((len(labels),1)), **kwargs)
