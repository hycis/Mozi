import argparse
import cPickle
import glob
import os
import numpy as np
import pynet.datasets.preprocessor as procs

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='receive epoch_error.pkl from cluster and plot it')
parser.add_argument('--cluster', metavar='CLUSTER', help='the name of cluster helios|biaree')
parser.add_argument('--model', metavar='NAME', help='name of the model')

args = parser.parse_args()

host_dir = "/Volumes/Storage/epoch_error"

gill='hycis@guillimin.clumeq.ca:/sb/project/jvb-000-aa/zhenzhou/Pynet/save/log'
biaree='hycis@briaree.calculquebec.ca:/RQexec/hycis/Pynet/save/log'
udem='wuzhen@frontal07.iro.umontreal.ca:~/Pynet/save/log'
nii='zhenzhou@136.187.97.216:~/Pynet/save/log'
helios='hycis@helios.calculquebec.ca:/scratch/jvb-000-aa/hycis/Pynet/save/log'


epoch_error_dir = "%s/%s"%(host_dir, args.model)
if not os.path.exists(epoch_error_dir):
    os.mkdir(epoch_error_dir)

def plot(epoch, train_error, valid_error, test_error):
    plt.xlabel("epoch")
    plt.ylabel("error")
    train, = plt.plot(epoch, train_error, label="train")
    valid, = plt.plot(epoch, valid_error, label="valid")
    test, = plt.plot(epoch, test_error, label="test")

    plt.legend([train, valid, test], ["train", "valid", "test"])

    plt.savefig("%s/plot.png"%epoch_error_dir)
    plt.show()



if args.cluster == "helios":
    if not os.path.exists(epoch_error_dir + "/epoch_error.pkl"):
        os.system('rsync -rvu %s/%s/epoch_error.pkl %s'%(helios, args.model, epoch_error_dir))
    fin = open('%s/epoch_error.pkl'%epoch_error_dir)
    error_epoch_tbl = cPickle.load(fin)
    plot(error_epoch_tbl['epoch'],
         error_epoch_tbl['train_error'],
         error_epoch_tbl['valid_error'],
         error_epoch_tbl['test_error'])

    fin.close()



elif args.cluster == "biaree":
    if not os.path.exists(epoch_error_dir + "/epoch_error.pkl"):
        os.system('rsync -rvu %s/%s/epoch_error.pkl %s'%(biaree, args.model, epoch_error_dir))
    fin = open('%s/epoch_error.pkl'%epoch_error_dir)
    error_epoch_tbl = cPickle.load(fin)
    plot(error_epoch_tbl['epoch'],
         error_epoch_tbl['train_error'],
         error_epoch_tbl['valid_error'],
         error_epoch_tbl['test_error'])

    fin.close()
