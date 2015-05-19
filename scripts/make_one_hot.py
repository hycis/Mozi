import argparse
import numpy as np
import os
import glob

parser = argparse.ArgumentParser(description='convert to one hot')
parser.add_argument('--input_data', metavar='PATH', help='path of the y dataset to be converted')
parser.add_argument('--dim', metavar='NUMBER', help='dimension of the one hot')
parser.add_argument('--save_dir', metavar='DIRECTORY', help='directory for the one hot')

args = parser.parse_args()

paths = glob.glob(args.input_data)
for path in paths:
    print 'opening..', path
    with open(path) as fin:
        x = np.load(fin)
    x_tmp = np.zeros((x.shape[0], int(args.dim)))
    for i in xrange(x.shape[0]):
        x_tmp[i, x[i]] = 1
    base = os.path.basename(path)
    print 'saving..', "%s/onehot_%s"%(args.save_dir, base)
    with open("%s/onehot_%s"%(args.save_dir, base), 'wb') as fout:
        np.save(fout, np.asarray(x_tmp))
    print

print 'done'
