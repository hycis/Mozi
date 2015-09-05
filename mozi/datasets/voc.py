

from mozi.utils.utils import get_file, make_one_hot
from mozi.datasets.dataset import SingleBlock
import xml.etree.ElementTree as ET
import os
import glob
from skimage.transform import resize
from skimage.io import imread
import cPickle
import marshal
import numpy as np

class VOC(SingleBlock):

    def __init__(self, resized_shape=(222,222,3), **kwargs):
        '''
        using only voc 2012 for actions classification, total 2154 images
        resized_shape is of (height, width, channel)
        '''
        im_dir = os.environ['MOZI_DATA_PATH'] + '/voc'
        path = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        im_dir = get_file(fpath="{}/VOCtrainval_11-May-2012.tar".format(im_dir), origin=path, untar=True)
        actls = ['jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike',
                   'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking',
                   'other']
        X_path = os.environ['MOZI_DATA_PATH'] + '/voc/X.npy'
        y_path = os.environ['MOZI_DATA_PATH'] + '/voc/y.npy'
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            print X_path + ' does not exists, generating..'
            annote = im_dir + '/VOC2012/Annotations'
            images = im_dir + '/VOC2012/JPEGImages'
            files = glob.glob(annote + '/2012*xml')
            labels = []
            rimage = []
            for f in files:
                bname = os.path.basename(f).rstrip('.xml')
                image = imread('{}/{}.jpg'.format(images, bname))
                rimage.append(resize(image, resized_shape))
                tree = ET.parse(f)
                root = tree.getroot()
                actions = root.find('object').find('actions')

                for act in actions:
                    if act.text == '1':
                        labels.append(actls.index(act.tag))
                        # only restrict to one action per photo
                        break

            print 'saving data'
            with open(X_path, 'wb') as Xout, open(y_path, 'wb') as yout:
                X = np.asarray(rimage)
                y = np.asarray(labels)
                np.save(Xout, X)
                np.save(yout, y)

        else:
            print X_path + ' exists, loading..'
            with open(X_path, 'rb') as Xin, open(y_path, 'rb') as yin:
                X = np.load(Xin)
                y = np.load(yin)

        super(VOC, self).__init__(X=np.rollaxis(X,3,1), y=make_one_hot(y,len(actls)), **kwargs)

# x = VOC()
