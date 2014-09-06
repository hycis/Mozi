import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='cut the frame of the dataset')
parser.add_argument('--dataset', metavar='PATH', help='path to the numpy data file')
parser.add_argument('--start', metavar='NUM', default=0, help='the starting index of the cut')
parser.add_argument('--end', metavar='NUM', default=-1, help='the ending index of the cut')
parser.add_argument('--output_dir', metavar='DIR', help='the output directory for cut dataset')
args = parser.parse_args()

paths = glob.glob(args.dataset)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for path in paths:
    print('..loading file %s'%path)
    fin = open(path)
    data = np.load(fin)

    print('..old shape ' + str(data.shape))
    cut = data[:, int(args.start):int(args.end)]
    print('..new shape ' + str(cut.shape))

    basename = os.path.basename(path)
    fout = open(args.output_dir + '/' + basename, 'wb')
    print('..save to ' + args.output_dir + '/' + basename)
    np.save(fout, cut)

    fin.close()
    fout.close()
